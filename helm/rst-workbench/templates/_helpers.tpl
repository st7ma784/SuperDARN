{{/*
Common labels applied to every resource in this chart.
*/}}
{{- define "rst-workbench.labels" -}}
app.kubernetes.io/name: rst-workbench
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: rst-workbench
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end -}}

{{/*
Resolve the cluster domain.

Per the ClusterOS helm-ingress guide, the source of truth is the
`clusteros-config` ConfigMap in kube-system (key `data.domain`), which the
node-agent always publishes from /etc/clusteros/cloudflare.env. The chart
reads it via Helm's lookup() — this is the ACTIVE path on every cluster.

`.Values.global.clusterDomain` takes precedence if set, so operators can
override via `--set global.clusterDomain=...` or re-enable Fleet's optional
`valuesFrom: clusteros-helm-values` path in fleet.yaml on clusters whose
node-agent also publishes that second ConfigMap.

The Rancher cluster-label lookup used by an earlier version of this chart
is intentionally retired.

Returns the bare domain (e.g. "example.com") or "" when the cluster has no
domain configured (nip.io mode).
*/}}
{{- define "rst-workbench.clusterDomain" -}}
{{- $fromValues := "" -}}
{{- if .Values.global -}}
  {{- $fromValues = .Values.global.clusterDomain | default "" -}}
{{- end -}}
{{- if $fromValues -}}
{{- $fromValues -}}
{{- else -}}
{{- $cm := lookup "v1" "ConfigMap" "kube-system" "clusteros-config" -}}
{{- if and $cm $cm.data -}}
{{- index $cm.data "domain" | default "" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Pick a node InternalIP for nip.io fallback. Used only when no cluster domain
is configured. Returns "" if no nodes are discoverable at template time.
*/}}
{{- define "rst-workbench.nodeIP" -}}
{{- $nodes := (lookup "v1" "Node" "" "").items -}}
{{- if $nodes -}}
{{- $ip := "" -}}
{{- range (first 1 $nodes) -}}
  {{- range .status.addresses -}}
    {{- if eq .type "InternalIP" -}}{{- $ip = .address -}}{{- end -}}
  {{- end -}}
{{- end -}}
{{- $ip -}}
{{- end -}}
{{- end -}}

{{/*
Resolve the effective ingress host.

Priority:
  1. Explicit value in .Values.ingress.hosts[0].host (escape hatch).
  2. clusteros-config ConfigMap domain -> <ingressSubdomain>.<domain>
  3. nip.io fallback -> <node-ip-with-dashes>.nip.io
     (LAN/Tailscale access only; cert-manager cannot issue for nip.io)

Returns "" only when none of the above resolve at template time, in which
case the ingress template skips rendering rules entirely.
*/}}
{{- define "rst-workbench.ingressHost" -}}
{{- $explicit := "" -}}
{{- if .Values.ingress.hosts -}}
  {{- $first := index .Values.ingress.hosts 0 -}}
  {{- if and $first $first.host -}}
    {{- $explicit = $first.host -}}
  {{- end -}}
{{- end -}}
{{- if $explicit -}}
{{- $explicit -}}
{{- else -}}
{{- $domain := include "rst-workbench.clusterDomain" . -}}
{{- if $domain -}}
{{- printf "%s.%s" .Values.ingressSubdomain $domain -}}
{{- else -}}
{{- $ip := include "rst-workbench.nodeIP" . -}}
{{- if $ip -}}
{{- printf "%s.nip.io" (replace "." "-" $ip) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Returns "domain" when a cluster domain is configured (or an explicit host is
set), otherwise "nip". Used to decide whether to emit a TLS block: Cloudflare
terminates TLS in domain mode, and nip.io has no issuable cert.
*/}}
{{- define "rst-workbench.ingressMode" -}}
{{- $explicit := "" -}}
{{- if .Values.ingress.hosts -}}
  {{- $first := index .Values.ingress.hosts 0 -}}
  {{- if and $first $first.host -}}{{- $explicit = $first.host -}}{{- end -}}
{{- end -}}
{{- if $explicit -}}domain
{{- else if include "rst-workbench.clusterDomain" . -}}domain
{{- else -}}nip
{{- end -}}
{{- end -}}
