{{- define "channel-emulator.fullname" -}}
channel-emulator
{{- end }}

{{- define "channel-emulator.labels" -}}
app.kubernetes.io/name: {{ include "channel-emulator.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: "{{ .Chart.AppVersion }}"
app.kubernetes.io/managed-by: Helm
{{- end }}

{{- define "channel-emulator.selectorLabels" -}}
app: channel-emulator
{{- end }}
