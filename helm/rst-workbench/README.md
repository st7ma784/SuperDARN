# rst-workbench Helm chart

Helm chart for the SuperDARN RST Interactive Workbench
(`external/rst/webapp/`): a FastAPI backend with selectable CPU and CUDA
variants, a React/Vite frontend, and an in-cluster Redis for Celery and
WebSocket pub/sub.

## What deploys

| Component       | Resources                                                   |
| --------------- | ----------------------------------------------------------- |
| `rst-backend`   | Deployment per enabled variant (`-cpu`, `-cuda`) sharing one Service |
| `rst-frontend`  | Deployment + Service                                        |
| `redis`         | StatefulSet + headless Service + ConfigMap + PVC            |
| Storage         | `rst-uploads` + `rst-results` PVCs (ReadWriteMany)          |
| Ingress         | Single host, path-routed: `/api`, `/ws`, `/docs` → backend; `/` → frontend |

Both backend variants run with `app=rst-backend` so they share the Service.
Disable one to force traffic onto the other; enable both to compare
CPU vs CUDA pipelines side-by-side.

## Ingress hostname

The host is resolved at template time via `lookup()` against the Rancher
cluster CRD (`management.cattle.io/v3 Cluster "local"`), reading the
`domain` label that Rancher stamps onto the cluster at provision time:

```
host = <ingressSubdomain>.<cluster.metadata.labels.domain>
```

So a cluster labelled `domain=stevemander.uk` lands the workbench at
`https://rst.stevemander.uk`. Nothing is hardcoded, and there is no Fleet
`${CLUSTER_LABELS_DOMAIN}` substitution to escape.

Override by setting `ingress.hosts[0].host` (or by changing
`ingressSubdomain`).

## Rancher Fleet

`fleet.yaml` lives in this directory, so Fleet treats the chart dir as the
bundle root and only packages `helm/rst-workbench/**`. A repo-level
`.fleetignore` keeps the GitRepo well clear of Rancher's 3 MB API request
limit by excluding `external/`, submodules, build outputs, notebooks and
model checkpoints — none of which belong in a Helm bundle.

## Quick install (without Fleet)

```bash
helm upgrade --install rst-workbench helm/rst-workbench \
  --create-namespace -n rst-workbench
```

Enable the CUDA backend on a GPU node:

```bash
helm upgrade --install rst-workbench helm/rst-workbench \
  --create-namespace -n rst-workbench \
  --set backend.cuda.enabled=true
```
