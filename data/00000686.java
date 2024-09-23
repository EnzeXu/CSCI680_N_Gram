public class PwEntryTestV3 { PwEntryV3 mPE; @Before public void setUp() throws Exception { Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext(); mPE = (PwEntryV3) TestData.GetTest1(ctx).entries.get(0); } @Test public void testName() { assertTrue("Name was " + mPE.title, mPE.title.equals("Amazon")); } @Test public void testPassword() throws UnsupportedEncodingException { String sPass = "12345"; byte[] password = sPass.getBytes("UTF-8"); assertArrayEquals(password, mPE.getPasswordBytes()); } @Test public void testCreation() { Calendar cal = Calendar.getInstance(); cal.setTime(mPE.tCreation.getJDate()); assertEquals("Incorrect year.", cal.get(Calendar.YEAR), 2009); assertEquals("Incorrect month.", cal.get(Calendar.MONTH), 3); assertEquals("Incorrect day.", cal.get(Calendar.DAY_OF_MONTH), 23); } }