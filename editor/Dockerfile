FROM golang:1.25-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /yamledit .

FROM alpine:3.20
RUN apk add --no-cache ca-certificates
COPY --from=build /yamledit /usr/local/bin/yamledit
WORKDIR /data
EXPOSE 8070
ENTRYPOINT ["yamledit", "-dir", "/data/index", "-port", "8070", "-no-browser"]
