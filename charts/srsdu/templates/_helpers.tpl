{{/*
Generate a full name for the release
*/}}
{{- define "srsdu.fullname" -}}
srsdu
{{- end }}

{{/*
Standard labels
*/}}
{{- define "srsdu.labels" -}}
app.kubernetes.io/name: {{ include "srsdu.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: "{{ .Chart.AppVersion }}"
app.kubernetes.io/managed-by: Helm
{{- end }}

{{/*
Selector labels
*/}}
{{- define "srsdu.selectorLabels" -}}
app: srsran
component: du
{{- end }}
