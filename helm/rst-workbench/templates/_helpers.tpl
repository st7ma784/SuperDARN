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
Resolve the effective ingress host.

Priority:
  1. Explicit value in .Values.ingress.hosts[0].host
  2. "domain" label on the Rancher local cluster CRD
     (management.cattle.io/v3 Cluster "local"), prefixed by
     .Values.ingressSubdomain.

Returns an empty string if neither source is available, in which case the
ingress template skips rendering rules.
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
{{- $domain := "" -}}
{{- $cluster := lookup "management.cattle.io/v3" "Cluster" "" "local" -}}
{{- if and $cluster $cluster.metadata $cluster.metadata.labels -}}
{{- $domain = index $cluster.metadata.labels "domain" | default "" -}}
{{- end -}}
{{- if $domain -}}
{{- printf "%s.%s" .Values.ingressSubdomain $domain -}}
{{- end -}}
{{- end -}}
{{- end -}}
