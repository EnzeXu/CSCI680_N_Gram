public class JUnit38SortingTest { private MaxCore fMax; private File fMaxFile; @Before public void createMax() { fMaxFile = new File("MaxCore.ser"); if (fMaxFile.exists()) { fMaxFile.delete(); } fMax = MaxCore.storedLocally(fMaxFile); } @After public void forgetMax() { fMaxFile.delete(); } public static class JUnit4Test { @Test public void pass() { } } public static class JUnit38Test extends TestCase { public void testFails() { fail(); } public void testSucceeds() { } public void testSucceedsToo() { } } @Test public void preferRecentlyFailed38Test() { Request request = Request.classes(JUnit4Test.class, JUnit38Test.class); fMax.run(request); List<Description> tests = fMax.sortedLeavesForTest(request); Description dontSucceed = Description.createTestDescription( JUnit38Test.class, "testFails"); assertEquals(dontSucceed, tests.get(0)); } }