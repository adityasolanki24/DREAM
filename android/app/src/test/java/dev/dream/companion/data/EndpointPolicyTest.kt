package dev.dream.companion.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class EndpointPolicyTest {
    @Test fun loopbackHttpIsAllowedForUsb() {
        assertNull(EndpointPolicy.check("http://127.0.0.1:8765"))
        assertNull(EndpointPolicy.check("http://localhost:8766/"))
        assertEquals("USB (adb reverse)", EndpointPolicy.kindOf("http://127.0.0.1:8765"))
    }

    @Test fun httpsIsAllowedAnywhere() {
        assertNull(EndpointPolicy.check("https://laptop.tail1234.ts.net"))
        assertNull(EndpointPolicy.check("https://robot.example.org:8443/dream"))
        assertEquals("Private tailnet (HTTPS)", EndpointPolicy.kindOf("https://laptop.tail1234.ts.net"))
    }

    @Test fun tailnetNamesAreAllowedOverWireGuard() {
        assertNull(EndpointPolicy.check("http://laptop-h2lk8p6d.tail6b0e06.ts.net:8765"))
        assertEquals("Private tailnet (WireGuard)", EndpointPolicy.kindOf("http://laptop-h2lk8p6d.tail6b0e06.ts.net:8765"))
        assertNotNull(EndpointPolicy.check("http://evil.ts.net.example.com:8765"))
    }

    @Test fun clearTextOverANetworkIsRefused() {
        assertNotNull(EndpointPolicy.check("http://100.70.120.48:8765"))
        assertNotNull(EndpointPolicy.check("http://192.168.1.20:8765"))
        assertNotNull(EndpointPolicy.check("ftp://robot"))
        assertNotNull(EndpointPolicy.check(""))
    }
}
