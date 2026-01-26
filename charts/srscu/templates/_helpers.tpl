{{/*
Generate a full name for the release
*/}}
{{- define "srscu.fullname" -}}
srscu
{{- end }}

{{/*
Standard labels
*/}}
{{- define "srscu.labels" -}}
app.kubernetes.io/name: {{ include "srscu.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: "{{ .Chart.AppVersion }}"
app.kubernetes.io/managed-by: Helm
{{- end }}

{{/*
Selector labels
*/}}
{{- define "srscu.selectorLabels" -}}
app: srsran
component: cu
{{- end }}
