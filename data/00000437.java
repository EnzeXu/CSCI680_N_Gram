public class SuiteMethodTest { public static boolean wasRun ; public static class OldTest extends TestCase { public OldTest ( String name ) { super ( name ) ; } public static junit . framework . Test suite ( ) { TestSuite suite = new TestSuite ( ) ; suite . addTest ( new OldTest ( "notObviouslyATest" ) ) ; return suite ; } public void notObviouslyATest ( ) { wasRun = true ; } } @ Test public void makeSureSuiteIsCalled ( ) { wasRun = false ; JUnitCore . runClasses ( OldTest . class ) ; assertTrue ( wasRun ) ; } public static class NewTest { @ Test public void sample ( ) { wasRun = true ; } public static junit . framework . Test suite ( ) { return new JUnit4TestAdapter ( NewTest . class ) ; } } @ Test public void makeSureSuiteWorksWithJUnit4Classes ( ) { wasRun = false ; JUnitCore . runClasses ( NewTest . class ) ; assertTrue ( wasRun ) ; } public static class CompatibilityTest { @ Ignore @ Test public void ignored ( ) { } public static junit . framework . Test suite ( ) { return new JUnit4TestAdapter ( CompatibilityTest . class ) ; } } @ Test public void descriptionAndRunNotificationsAreConsistent ( ) { Result result = JUnitCore . runClasses ( CompatibilityTest . class ) ; assertEquals ( 0 , result . getIgnoreCount ( ) ) ; Description description = Request . aClass ( CompatibilityTest . class ) . getRunner ( ) . getDescription ( ) ; assertEquals ( 0 , description . getChildren ( ) . size ( ) ) ; } public static class NewTestSuiteFails { @ Test public void sample ( ) { wasRun = true ; } public static junit . framework . Test suite ( ) { fail ( "called with JUnit 4 runner" ) ; return null ; } } @ Test public void suiteIsUsedWithJUnit4Classes ( ) { wasRun = false ; Result result = JUnitCore . runClasses ( NewTestSuiteFails . class ) ; assertEquals ( 1 , result . getFailureCount ( ) ) ; assertFalse ( wasRun ) ; } public static class NewTestSuiteNotUsed { private static boolean wasIgnoredRun ; @ Test public void sample ( ) { wasRun = true ; } @ Ignore @ Test public void ignore ( ) { wasIgnoredRun = true ; } public static junit . framework . Test suite ( ) { return new JUnit4TestAdapter ( NewTestSuiteNotUsed . class ) ; } } @ Test public void makeSureSuiteNotUsedWithJUnit4Classes2 ( ) { wasRun = false ; NewTestSuiteNotUsed . wasIgnoredRun = false ; Result res = JUnitCore . runClasses ( NewTestSuiteNotUsed . class ) ; assertTrue ( wasRun ) ; assertFalse ( NewTestSuiteNotUsed . wasIgnoredRun ) ; assertEquals ( 0 , res . getFailureCount ( ) ) ; assertEquals ( 1 , res . getRunCount ( ) ) ; assertEquals ( 0 , res . getIgnoreCount ( ) ) ; } }