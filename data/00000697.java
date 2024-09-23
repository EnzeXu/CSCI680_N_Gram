public class SprEngineTest { private PwDatabaseV4 db; private SprEngine spr; @Before public void setUp() throws Exception { Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext(); AssetManager am = ctx.getAssets(); InputStream is = am.open("test.kdbx", AssetManager.ACCESS_STREAMING); ImporterV4 importer = new ImporterV4(ctx.getFilesDir()); db = importer.openDatabase(is, "12345", null); is.close(); spr = SprEngine.getInstance(db); } private final String REF = "{REF:P@I:2B1D56590D961F48A8CE8C392CE6CD35}"; private final String ENCODE_UUID = "IN7RkON49Ui1UZ2ddqmLcw=="; private final String RESULT = "Password"; @Test public void testRefReplace() { UUID entryUUID = decodeUUID(ENCODE_UUID); PwEntryV4 entry = (PwEntryV4) db.entries.get(entryUUID); assertEquals(RESULT, spr.compile(REF, entry, db)); } private UUID decodeUUID(String encoded) { if (encoded == null || encoded.length() == 0 ) { return PwDatabaseV4.UUID_ZERO; } byte[] buf = Base64.decode(encoded, Base64.NO_WRAP); return Types.bytestoUUID(buf); } }