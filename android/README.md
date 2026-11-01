# DREAM companion (Android)

Native Kotlin/Compose client for DREAM's phone API. Android 9 (API 28) minimum; verified on an OPPO CPH2015.
It sends goals (typed or push-to-talk), shows the robot's camera, map and memory, and pauses or cancels tasks.
Robots execute on their own DREAM runtime; the phone never commands motion directly.

## Build and install

```bash
cd android
./gradlew :app:testDebugUnitTest :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

`local.properties` needs `sdk.dir`; JDK 17+.

## Connecting to a robot

Settings → **Robots** holds any number of saved connections. Each has a name, an endpoint and its own access token,
encrypted with the Android Keystore and sent only as an `Authorization` header. The endpoint decides the transport:

| Where the robot is | Endpoint | Notes |
| --- | --- | --- |
| Laptop, phone on USB | `http://127.0.0.1:8765` | Run `adb reverse tcp:8765 tcp:8765`. Re-run after the cable is re-plugged. |
| Anywhere on your tailnet | `http://<machine>.<tailnet>.ts.net:<port>` | Phone and robot both on Tailscale; MagicDNS names resolve only inside the tailnet and traffic is WireGuard-encrypted. On the robot machine, publish the API to the tailnet only, for example `tailscale serve --bg --tcp <port> tcp://127.0.0.1:8765`. |
| A robot with its own TLS | `https://host[:port]` | Any HTTPS endpoint, such as an on-board server or a reverse proxy. |

Plain HTTP to a raw IP address over a network is refused (`EndpointPolicy` in `data/RobotLink.kt`, mirrored by
`res/xml/network_security_config.xml`). Never expose the phone API to the public internet.

The app reconnects automatically when a network returns and backs off (1.5 s to 10 s) while a robot is unreachable.

## Making another robot work

Everything above the network goes through one interface, `RobotLink` (`data/RobotLink.kt`). `DreamApi` implements it
for DREAM's phone API v1 (`dream/runtime/phone_api.py`):

| Endpoint | Purpose | Required |
| --- | --- | --- |
| `GET /status`, `GET /tasks`, `GET /snapshot` | robot state, task scheduler, providers, map, memory | yes |
| `GET /ag-ui/events?after=n` | AG-UI tool and run events | yes |
| `POST /tasks` | submit a goal with the selected provider and model | yes |
| `POST /tasks/{id}/pause\|resume\|cancel` | interruption at the scheduler's safe boundary | yes |
| `GET /camera/stream` (MJPEG), `GET /camera/frame?after=` | head camera | optional |
| `GET /camera/meta`, `GET /media/status` | camera timing, WebRTC availability | optional |

A robot whose runtime serves these endpoints works without app changes; optional features it lacks are detected per
connection (`capabilities()`) and shown as unavailable rather than faked. A robot with a different protocol needs only
another `RobotLink` implementation, chosen in `Session.linkFor`.
