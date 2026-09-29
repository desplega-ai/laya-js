# AWS

**Status: untested.** Nothing here has been deployed. Limits checked 2026-09-29.

Push the image to ECR first:

```sh
aws ecr create-repository --repository-name laya-server
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker tag laya-server:local <acct>.dkr.ecr.<region>.amazonaws.com/laya-server:<tag>
docker push <acct>.dkr.ecr.<region>.amazonaws.com/laya-server:<tag>
aws secretsmanager create-secret --name laya/api-key --secret-string "$(openssl rand -hex 16)"
```

## ECS Fargate

**Verdict: works.** Nothing binds.

| Limit | Value | Source |
| --- | --- | --- |
| Task sizes | 1 vCPU: 2 to 8 GB; 2 vCPU: 4 to 16 GB; 4 vCPU: 8 to 30 GB | [Fargate task sizes](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-tasks-services.html) |
| Image size | no stated limit; the image counts against ephemeral storage, 20 GiB default, up to 200 GiB | [Fargate task storage](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-task-storage.html) |
| Container health check `startPeriod` | 0 to 300 s | [task definition parameters](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definition_parameters.html) |

Task definition (one checkpoint, 1 vCPU / 4 GB):

```json
{
  "family": "laya-server",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "1024",
  "memory": "4096",
  "runtimePlatform": { "cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX" },
  "executionRoleArn": "arn:aws:iam::<acct>:role/ecsTaskExecutionRole",
  "containerDefinitions": [{
    "name": "laya-server",
    "image": "<acct>.dkr.ecr.<region>.amazonaws.com/laya-server:<tag>",
    "portMappings": [{ "containerPort": 8000 }],
    "environment": [{ "name": "LAYA_THREADS", "value": "1" }],
    "secrets": [{ "name": "LAYA_API_KEY", "valueFrom": "arn:aws:secretsmanager:<region>:<acct>:secret:laya/api-key" }],
    "healthCheck": {
      "command": ["CMD", "node", "-e", "fetch('http://127.0.0.1:8000/health').then(r=>process.exit(r.status===200?0:1),()=>process.exit(1))"],
      "interval": 15, "timeout": 5, "retries": 3, "startPeriod": 300
    },
    "stopTimeout": 30,
    "logConfiguration": { "logDriver": "awslogs", "options": {
      "awslogs-group": "/ecs/laya-server", "awslogs-region": "<region>",
      "awslogs-stream-prefix": "laya", "awslogs-create-group": "true" } }
  }]
}
```

The image has no `curl`, so the health check uses Node's `fetch`. Put an ALB target group in front with health check path `/health` and set the service's `healthCheckGracePeriodSeconds` to 300.

```sh
aws ecs register-task-definition --cli-input-json file://task-def.json
aws ecs create-service --cluster <cluster> --service-name laya-server \
  --task-definition laya-server --desired-count 2 --launch-type FARGATE \
  --health-check-grace-period-seconds 300 \
  --network-configuration 'awsvpcConfiguration={subnets=[<subnet>],securityGroups=[<sg>]}' \
  --load-balancers 'targetGroupArn=<tg-arn>,containerName=laya-server,containerPort=8000'
```

For all three checkpoints: `"cpu": "2048"`, `"memory": "8192"`, `LAYA_MODELS=multilingual,english,typed-decisions`, an `HF_TOKEN` secret and `"ephemeralStorage": { "sizeInGiB": 30 }`.

## Lambda (container image)

**Verdict: works with caveats.** The binding constraint is cold start.

| Limit | Value | Source |
| --- | --- | --- |
| Container image | 10 GB uncompressed | [Lambda quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html) |
| Memory | 128 to 10,240 MB; 1,769 MB is one vCPU | same |
| Timeout | 900 s | same |
| Sync payload | 6 MB request and response | same |
| Init phase | "limited to 10 seconds"; on timeout, init is retried inside the first invocation under the function timeout | [execution environment](https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtime-environment.html) |

The model load takes about 7 s with the file in the page cache; a fresh Lambda environment reading a 1.6 GB image will likely exceed the 10 s init budget, so the first request on each new environment pays the full load. Provisioned concurrency avoids that at a fixed hourly cost. The filesystem is read-only except `/tmp`.

Wrap the image with the [Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter):

```dockerfile
# Dockerfile.lambda
FROM <acct>.dkr.ecr.<region>.amazonaws.com/laya-server:<tag>
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter
ENV AWS_LWA_PORT=8000 \
    AWS_LWA_READINESS_CHECK_PATH=/health \
    LAYA_CACHE_DIR=/tmp/cache \
    LAYA_THREADS=2
```

```sh
docker build -f Dockerfile.lambda -t <acct>.dkr.ecr.<region>.amazonaws.com/laya-server:<tag>-lambda .
docker push <acct>.dkr.ecr.<region>.amazonaws.com/laya-server:<tag>-lambda
aws lambda create-function --function-name laya-server --package-type Image \
  --code ImageUri=<acct>.dkr.ecr.<region>.amazonaws.com/laya-server:<tag>-lambda \
  --role arn:aws:iam::<acct>:role/laya-lambda --memory-size 4096 --timeout 60 \
  --environment 'Variables={LAYA_API_KEY=<from Secrets Manager or SSM at deploy time>}'
aws lambda create-function-url-config --function-name laya-server --auth-type AWS_IAM
```

`--memory-size 4096` gives about 2.3 vCPU. Lambda does not honour the image `HEALTHCHECK`; the adapter polls `/health` before forwarding the first request.

## App Runner

**Verdict: not viable for new accounts.** "AWS App Runner is no longer open to new customers. Existing customers can continue to use the service as normal" ([availability change](https://docs.aws.amazon.com/apprunner/latest/dg/apprunner-availability-change.html), checked 2026-09-29). AWS points new users at ECS Express Mode, which runs on Fargate: use the task definition above. Existing App Runner accounts can deploy the image with 1 vCPU / 4 GB, port 8000 and an HTTP health check on `/health`.
