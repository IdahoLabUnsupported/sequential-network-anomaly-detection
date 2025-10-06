## HTTP Log
WEBSHELL attack
Log4j attack
#### New Fields
- [x] safe_port (80, 8080, 8088, 8000)
- [x] file_sent (is the mime type not NaN)
- [x] successful (status code was 200 or 404)
- [x] sus_user_agent (is the user agent one of [curl, wget, python-urllib, powershell])

#### Modified Fields
- [x] proxied (binary)
- [x] response_body_len (min max scaled)
- [x] request_body_len

#### Removed Fields
All other fields not listeds above are removed (with the exception of ts and uid)