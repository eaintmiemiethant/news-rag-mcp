#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../infra/cdk"
npm install
npx cdk bootstrap
npx cdk deploy --all
