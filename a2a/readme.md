```bash
export SERVICE_NAME="reporter"
export REPO_NAME="a2a"
export PROJECT_ID="vtxdemos"
export REGION="us-central1"
export IMAGE_TAG="latest"
export AGENT_NAME="reporter"
export IMAGE_NAME="reporter-agent"
export PUBLIC_URL="https://reporter-agent-${PROJECT_NUMBER}.${REGION}.run.app"
export IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
```

```bash
echo "Building ${AGENT_NAME} agent..."
gcloud builds submit . \
  --config=cloudbuild-build.yaml \
  --project=${PROJECT_ID} \
  --region=${REGION} \
  --substitutions=_AGENT_NAME=${AGENT_NAME},_REGION=${REGION},_REPO_NAME=${REPO_NAME},_PROJECT_ID=${PROJECT_ID},_IMAGE_TAG=${IMAGE_TAG}

echo "Image built and pushed to: ${IMAGE_PATH}"
```

```bash
gcloud run deploy ${SERVICE_NAME} \
  --image=${IMAGE_PATH} \
  --platform=managed \
  --region=${REGION} \
  --set-env-vars="A2A_HOST=0.0.0.0" \
  --set-env-vars="A2A_PORT=8080" \
  --set-env-vars="GOOGLE_GENAI_USE_VERTEXAI=TRUE" \
  --set-env-vars="GOOGLE_CLOUD_LOCATION=${REGION}" \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
  --set-env-vars="PUBLIC_URL=${PUBLIC_URL}" \
  --allow-unauthenticated \
  --project=${PROJECT_ID} \
  --min-instances=1
```