#!/usr/bin/env bash
# Stand-in for the AWS CLI in test_deploy_dry_run.sh: canned answers for an account without a flybot stack.
# Every call is appended to $FAKE_LOG.
printf '%s\n' "$*" >> "${FAKE_LOG:-/dev/null}"
a=" $* "
case "$a" in
  *" --version "*) echo "aws-cli/2.99.0 Python/3.12 fake"; exit 0 ;;
  *" sts get-caller-identity "*"--query Account "*) echo 123456789012 ;;
  *" sts get-caller-identity "*"--query Arn "*) echo "arn:aws:sts::123456789012:assumed-role/TestRole/session" ;;
  *" sts get-caller-identity "*) echo '{"Account":"123456789012"}' ;;
  *" ec2 describe-availability-zones "*) printf 'ap-southeast-2a\tap-southeast-2b\tap-southeast-2c\n' ;;
  *" ec2 describe-vpcs "*) echo vpc-0fake ;;
  *" ec2 describe-instance-type-offerings "*) printf 'ap-southeast-2a\tap-southeast-2b\n' ;;
  *" service-quotas get-service-quota "*"L-0263D0A3"*) echo 5.0 ;;
  *" service-quotas get-service-quota "*) echo 32.0 ;;
  *" ec2 describe-addresses "*) echo 1 ;;
  *" ec2 describe-instances "*) printf '1\t2\n' ;;
  *" ec2 describe-images "*) echo ami-0fake0000000000000 ;;
  *" iot describe-endpoint "*"--query endpointAddress "*) echo fake123-ats.iot.ap-southeast-2.amazonaws.com ;;
  *" cloudformation describe-stacks "*|*" cloudformation describe-stack-resource "*)
     echo "An error occurred (ValidationError) when calling the DescribeStacks operation: Stack with id flybot-citest does not exist" >&2; exit 254 ;;
  *" sns list-topics "*"--query "*) printf 'arn:aws:sns:ap-southeast-2:123456789012:sentinel-dev-alarms\tarn:aws:sns:ap-southeast-2:123456789012:other\n' ;;
  *" sns list-subscriptions-by-topic "*"sentinel-dev-alarms"*) echo owner@example.com ;;
  *" sns list-subscriptions-by-topic "*) echo "" ;;
  *" s3api head-bucket "*"models-public"*) exit 0 ;;
  *" s3api head-bucket "*) echo "An error occurred (404) when calling the HeadBucket operation: Not Found" >&2; exit 254 ;;
  *" s3api head-object "*) echo "An error occurred (404)" >&2; exit 254 ;;
  *" secretsmanager describe-secret "*"sentinel-dev-anthropic"*) echo None ;;
  *" secretsmanager describe-secret "*) echo "An error occurred (ResourceNotFoundException)" >&2; exit 254 ;;
  *" ecr describe-images "*) echo "An error occurred (RepositoryNotFoundException)" >&2; exit 254 ;;
  *) : ;;   # the read-only API probes (list-*, describe-*) answer with nothing
esac
exit 0
