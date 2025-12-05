# Deployment Guide

## Overview

This guide provides comprehensive strategies for deploying Theta Sketch Decision Tree classifiers in production environments, from single-instance deployments to scalable cloud architectures.

## Deployment Strategies

### 1. Single Instance Deployment

#### Local Server Deployment

```python
from flask import Flask, request, jsonify
from theta_sketch_tree.model_persistence import ModelPersistence
import numpy as np

app = Flask(__name__)

# Load model at startup
model = ModelPersistence.load_model('production_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """REST API endpoint for predictions."""
    try:
        # Parse request
        data = request.get_json()
        features = np.array(data['features'], dtype=int)

        # Validate input
        if features.shape[1] != model.n_features_:
            return jsonify({'error': f'Expected {model.n_features_} features, got {features.shape[1]}'}), 400

        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        return jsonify({
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
```

#### Docker Containerization

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip install -e .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "30", "app:app"]
```

### 2. Microservice Architecture

#### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: theta-sketch-classifier
  labels:
    app: theta-sketch-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: theta-sketch-classifier
  template:
    metadata:
      labels:
        app: theta-sketch-classifier
    spec:
      containers:
      - name: classifier
        image: your-registry/theta-sketch-classifier:latest
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_PATH
          value: "/app/models/production_model.pkl"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: classifier-service
spec:
  selector:
    app: theta-sketch-classifier
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: classifier-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: classifier.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: classifier-service
            port:
              number: 80
```

### 3. Serverless Deployment

#### AWS Lambda

```python
import json
import boto3
import numpy as np
from theta_sketch_tree.model_persistence import ModelPersistence

# Load model at module level (cold start optimization)
s3 = boto3.client('s3')
model = None

def load_model_from_s3():
    """Load model from S3 storage."""
    global model
    if model is None:
        # Download model from S3
        s3.download_file('your-model-bucket', 'production_model.pkl', '/tmp/model.pkl')
        model = ModelPersistence.load_model('/tmp/model.pkl')
    return model

def lambda_handler(event, context):
    """AWS Lambda handler for predictions."""
    try:
        # Load model
        clf = load_model_from_s3()

        # Parse input
        body = json.loads(event['body'])
        features = np.array(body['features'], dtype=int)

        # Validate input
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != clf.n_features_:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'Expected {clf.n_features_} features'})
            }

        # Make predictions
        predictions = clf.predict(features)
        probabilities = clf.predict_proba(features)

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist()
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

#### Lambda Configuration (SAM Template)

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  ClassifierFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: classifier/
      Handler: lambda_function.lambda_handler
      Runtime: python3.9
      Timeout: 30
      MemorySize: 1024
      Environment:
        Variables:
          MODEL_BUCKET: !Ref ModelBucket
      Policies:
        - S3ReadPolicy:
            BucketName: !Ref ModelBucket
      Events:
        ClassifyApi:
          Type: Api
          Properties:
            Path: /predict
            Method: post

  ModelBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: theta-sketch-models

Outputs:
  ClassifierApi:
    Description: "API Gateway endpoint URL"
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/predict/"
```

## Performance Optimization

### 1. Model Optimization

#### Model Preprocessing

```python
class OptimizedClassifier:
    """Optimized classifier for production deployment."""

    def __init__(self, base_model):
        self.base_model = base_model
        self.feature_names = base_model.feature_names_
        self.n_features = base_model.n_features_
        self._optimize_tree_structure()

    def _optimize_tree_structure(self):
        """Convert tree to optimized format for fast traversal."""
        # Convert to flat array representation for cache efficiency
        self.tree_array = self._flatten_tree(self.base_model.tree_)

    def _flatten_tree(self, node, index=0):
        """Convert tree to flat array representation."""
        # Implementation would create compact tree representation
        # for optimal cache performance
        pass

    def predict_optimized(self, X):
        """Optimized prediction using flat tree structure."""
        # Use vectorized operations and cache-friendly traversal
        return self._traverse_flat_tree(X)

# Usage in production
def create_optimized_model(model_path):
    """Create production-optimized model."""
    base_model = ModelPersistence.load_model(model_path)
    optimized_model = OptimizedClassifier(base_model)
    return optimized_model
```

### 2. Caching Strategies

#### Redis Caching

```python
import redis
import pickle
import hashlib
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_predictions(expiration=3600):
    """Decorator to cache predictions."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, X):
            # Create cache key from input hash
            input_hash = hashlib.md5(X.tobytes()).hexdigest()
            cache_key = f"prediction:{input_hash}"

            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return pickle.loads(cached_result)

            # Compute prediction
            result = func(self, X)

            # Cache result
            redis_client.setex(
                cache_key,
                expiration,
                pickle.dumps(result)
            )

            return result
        return wrapper
    return decorator

class CachedClassifier:
    """Classifier with prediction caching."""

    def __init__(self, model):
        self.model = model

    @cache_predictions(expiration=1800)  # 30 minutes
    def predict(self, X):
        """Cached prediction method."""
        return self.model.predict(X)

    @cache_predictions(expiration=1800)
    def predict_proba(self, X):
        """Cached probability prediction."""
        return self.model.predict_proba(X)
```

### 3. Load Balancing and Scaling

#### Auto-scaling Configuration

```python
# Kubernetes HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: classifier-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: theta-sketch-classifier
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

## Monitoring and Observability

### 1. Application Metrics

#### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Define metrics
PREDICTION_REQUESTS = Counter('predictions_total', 'Total prediction requests')
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
ERROR_RATE = Counter('prediction_errors_total', 'Total prediction errors')

class MonitoredClassifier:
    """Classifier with monitoring instrumentation."""

    def __init__(self, model):
        self.model = model
        self.recent_accuracies = []

    def predict_with_monitoring(self, X):
        """Prediction with monitoring metrics."""
        PREDICTION_REQUESTS.inc()

        start_time = time.time()
        try:
            # Make prediction
            predictions = self.model.predict(X)

            # Record latency
            latency = time.time() - start_time
            PREDICTION_LATENCY.observe(latency)

            return predictions

        except Exception as e:
            ERROR_RATE.inc()
            raise

    def update_accuracy_metric(self, accuracy):
        """Update accuracy tracking."""
        self.recent_accuracies.append(accuracy)
        if len(self.recent_accuracies) > 100:
            self.recent_accuracies.pop(0)

        current_accuracy = sum(self.recent_accuracies) / len(self.recent_accuracies)
        MODEL_ACCURACY.set(current_accuracy)

# Metrics endpoint
@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': 'text/plain'}
```

### 2. Logging Configuration

```python
import logging
import structlog
from datetime import datetime

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class LoggedClassifier:
    """Classifier with comprehensive logging."""

    def __init__(self, model, model_version="1.0"):
        self.model = model
        self.model_version = model_version

    def predict_with_logging(self, X, request_id=None):
        """Prediction with detailed logging."""
        start_time = datetime.utcnow()

        logger.info(
            "prediction_started",
            request_id=request_id,
            model_version=self.model_version,
            input_shape=X.shape,
            timestamp=start_time.isoformat()
        )

        try:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)

            end_time = datetime.utcnow()
            latency = (end_time - start_time).total_seconds()

            logger.info(
                "prediction_completed",
                request_id=request_id,
                model_version=self.model_version,
                latency_seconds=latency,
                predictions_count=len(predictions),
                timestamp=end_time.isoformat()
            )

            return predictions, probabilities

        except Exception as e:
            logger.error(
                "prediction_failed",
                request_id=request_id,
                model_version=self.model_version,
                error=str(e),
                error_type=type(e).__name__,
                timestamp=datetime.utcnow().isoformat()
            )
            raise
```

## Security Considerations

### 1. Input Validation

```python
from cerberus import Validator
import numpy as np

class SecureClassifier:
    """Classifier with security validation."""

    def __init__(self, model):
        self.model = model
        self.input_validator = Validator({
            'features': {
                'type': 'list',
                'required': True,
                'schema': {
                    'type': 'list',
                    'schema': {'type': 'integer', 'min': -1, 'max': 1}
                }
            }
        })

    def secure_predict(self, request_data):
        """Prediction with security validation."""
        # Validate input schema
        if not self.input_validator.validate(request_data):
            raise ValueError(f"Invalid input: {self.input_validator.errors}")

        features = np.array(request_data['features'], dtype=int)

        # Additional security checks
        if features.size > 100000:  # Prevent memory exhaustion
            raise ValueError("Input too large")

        if features.shape[1] != self.model.n_features_:
            raise ValueError("Feature count mismatch")

        # Rate limiting could be implemented here
        return self.model.predict(features)
```

### 2. Model Protection

```python
import hashlib
import hmac

class ProtectedModel:
    """Model with integrity protection."""

    def __init__(self, model_path, secret_key):
        self.model = ModelPersistence.load_model(model_path)
        self.secret_key = secret_key
        self.model_hash = self._compute_model_hash()

    def _compute_model_hash(self):
        """Compute model integrity hash."""
        # Create hash of model structure and parameters
        model_bytes = pickle.dumps(self.model.tree_)
        return hmac.new(
            self.secret_key.encode(),
            model_bytes,
            hashlib.sha256
        ).hexdigest()

    def verify_model_integrity(self):
        """Verify model hasn't been tampered with."""
        current_hash = self._compute_model_hash()
        return hmac.compare_digest(self.model_hash, current_hash)

    def secure_predict(self, X):
        """Prediction with integrity verification."""
        if not self.verify_model_integrity():
            raise SecurityError("Model integrity check failed")

        return self.model.predict(X)
```

## CI/CD Pipeline

### 1. Model Deployment Pipeline

```yaml
# .github/workflows/model-deployment.yml
name: Model Deployment

on:
  push:
    paths:
      - 'models/**'
    branches:
      - main

jobs:
  validate-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .

      - name: Validate model
        run: |
          python scripts/validate_model.py models/production_model.pkl

      - name: Run model tests
        run: |
          pytest tests/test_model_deployment.py

  deploy-staging:
    needs: validate-model
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/classifier-staging \
            classifier=your-registry/theta-sketch-classifier:${{ github.sha }}

      - name: Run integration tests
        run: |
          pytest tests/test_integration_staging.py

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/classifier-prod \
            classifier=your-registry/theta-sketch-classifier:${{ github.sha }} \
            --record

      - name: Verify deployment
        run: |
          kubectl rollout status deployment/classifier-prod
          python scripts/health_check.py https://classifier.example.com
```

### 2. Model Versioning

```python
class ModelRegistry:
    """Model registry for version management."""

    def __init__(self, storage_backend):
        self.storage = storage_backend

    def register_model(self, model, version, metadata):
        """Register new model version."""
        model_info = {
            'version': version,
            'created_at': datetime.utcnow().isoformat(),
            'metadata': metadata,
            'performance': self._evaluate_model(model)
        }

        # Save model and metadata
        model_path = f"models/v{version}/model.pkl"
        metadata_path = f"models/v{version}/metadata.json"

        self.storage.save_model(model, model_path)
        self.storage.save_json(model_info, metadata_path)

        return model_info

    def get_latest_model(self):
        """Get latest production model."""
        versions = self.storage.list_versions()
        latest_version = max(versions)
        return self.storage.load_model(f"models/v{latest_version}/model.pkl")

    def rollback_model(self, target_version):
        """Rollback to specific model version."""
        model_path = f"models/v{target_version}/model.pkl"
        if not self.storage.exists(model_path):
            raise ValueError(f"Model version {target_version} not found")

        return self.storage.load_model(model_path)
```

---

## Next Steps

- **Performance**: See [Performance Guide](06-performance.md) for optimization techniques
- **Monitoring**: Review monitoring setup and alerting strategies
- **Security**: Implement additional security measures as needed
- **Troubleshooting**: Check [Troubleshooting Guide](08-troubleshooting.md) for deployment issues