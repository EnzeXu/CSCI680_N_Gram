public class AccentTest { private static final String KEYFILE = ""; private static final String PASSWORD = "é"; private static final String ASSET = "accent.kdb"; private static final String FILENAME = "/sdcard/accent.kdb"; @Test public void testOpen() { try { Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext(); TestData.GetDb(ctx, ASSET, PASSWORD, KEYFILE, FILENAME); } catch (Exception e) { assertTrue("Failed to open database", false); } } }