public class MethodRulesTest { private static boolean ruleWasEvaluated ; private static class TestMethodRule implements MethodRule { public Statement apply ( final Statement base , FrameworkMethod method , Object target ) { return new Statement ( ) { @ Override public void evaluate ( ) throws Throwable { ruleWasEvaluated = true ; base . evaluate ( ) ; } } ; } } public static class ExampleTest { @ Rule public MethodRule example = new TestMethodRule ( ) ; @ Test public void nothing ( ) { } } abstract static class NonPublicExampleTest { @ Rule public MethodRule example = new TestMethodRule ( ) ; @ Test public void nothing ( ) { } } @ Test public void ruleIsIntroducedAndEvaluated ( ) { ruleWasEvaluated = false ; assertThat ( testResult ( ExampleTest . class ) , isSuccessful ( ) ) ; assertTrue ( ruleWasEvaluated ) ; } public static class SonOfExampleTest extends ExampleTest { } @ Test public void ruleIsIntroducedAndEvaluatedOnSubclass ( ) { ruleWasEvaluated = false ; assertThat ( testResult ( SonOfExampleTest . class ) , isSuccessful ( ) ) ; assertTrue ( ruleWasEvaluated ) ; } public static class SonOfNonPublicExampleTest extends NonPublicExampleTest { } @ Test public void ruleIsIntroducedAndEvaluatedOnSubclassOfNonPublicClass ( ) { ruleWasEvaluated = false ; assertThat ( testResult ( SonOfNonPublicExampleTest . class ) , isSuccessful ( ) ) ; assertTrue ( ruleWasEvaluated ) ; } private static int runCount ; private static class Increment implements MethodRule { public Statement apply ( final Statement base , FrameworkMethod method , Object target ) { return new Statement ( ) { @ Override public void evaluate ( ) throws Throwable { runCount++ ; base . evaluate ( ) ; } } ; } } public static class MultipleRuleTest { @ Rule public MethodRule incrementor1 = new Increment ( ) ; @ Rule public MethodRule incrementor2 = new Increment ( ) ; @ Test public void nothing ( ) { } } @ Test public void multipleRulesAreRun ( ) { runCount = 0 ; assertThat ( testResult ( MultipleRuleTest . class ) , isSuccessful ( ) ) ; assertEquals ( 2 , runCount ) ; } public static class NoRulesTest { public int x ; @ Test public void nothing ( ) { } } @ Test public void ignoreNonRules ( ) { assertThat ( testResult ( NoRulesTest . class ) , isSuccessful ( ) ) ; } private static String log ; public static class OnFailureTest { @ Rule public MethodRule watchman = new TestWatchman ( ) { @ Override public void failed ( Throwable e , FrameworkMethod method ) { log += method . getName ( ) + " " + e . getClass ( ) . getSimpleName ( ) ; } } ; @ Test public void nothing ( ) { fail ( ) ; } } @ Test public void onFailure ( ) { log = "" ; assertThat ( testResult ( OnFailureTest . class ) , failureCountIs ( 1 ) ) ; assertEquals ( "nothing AssertionError" , log ) ; } public static class WatchmanTest { private static String watchedLog ; @ Rule public MethodRule watchman = new TestWatchman ( ) { @ Override public void failed ( Throwable e , FrameworkMethod method ) { watchedLog += method . getName ( ) + " " + e . getClass ( ) . getSimpleName ( ) + "\n" ; } @ Override public void succeeded ( FrameworkMethod method ) { watchedLog += method . getName ( ) + " " + "success!\n" ; } } ; @ Test public void fails ( ) { fail ( ) ; } @ Test public void succeeds ( ) { } } @ Test public void succeeded ( ) { WatchmanTest . watchedLog = "" ; assertThat ( testResult ( WatchmanTest . class ) , failureCountIs ( 1 ) ) ; assertThat ( WatchmanTest . watchedLog , containsString ( "fails AssertionError" ) ) ; assertThat ( WatchmanTest . watchedLog , containsString ( "succeeds success!" ) ) ; } public static class BeforesAndAfters { private static String watchedLog ; @ Before public void before ( ) { watchedLog += "before " ; } @ Rule public MethodRule watchman = new TestWatchman ( ) { @ Override public void starting ( FrameworkMethod method ) { watchedLog += "starting " ; } @ Override public void finished ( FrameworkMethod method ) { watchedLog += "finished " ; } @ Override public void succeeded ( FrameworkMethod method ) { watchedLog += "succeeded " ; } } ; @ After public void after ( ) { watchedLog += "after " ; } @ Test public void succeeds ( ) { watchedLog += "test " ; } } @ Test public void beforesAndAfters ( ) { BeforesAndAfters . watchedLog = "" ; assertThat ( testResult ( BeforesAndAfters . class ) , isSuccessful ( ) ) ; assertThat ( BeforesAndAfters . watchedLog , is ( "starting before test after succeeded finished " ) ) ; } public static class WrongTypedField { @ Rule public int x = 5 ; @ Test public void foo ( ) { } } @ Test public void validateWrongTypedField ( ) { assertThat ( testResult ( WrongTypedField . class ) , hasSingleFailureContaining ( "must implement MethodRule" ) ) ; } public static class SonOfWrongTypedField extends WrongTypedField { } @ Test public void validateWrongTypedFieldInSuperclass ( ) { assertThat ( testResult ( SonOfWrongTypedField . class ) , hasSingleFailureContaining ( "must implement MethodRule" ) ) ; } public static class PrivateRule { @ Rule private TestRule rule = new TestName ( ) ; @ Test public void foo ( ) { } } @ Test public void validatePrivateRule ( ) { assertThat ( testResult ( PrivateRule . class ) , hasSingleFailureContaining ( "must be public" ) ) ; } public static class CustomTestName implements TestRule { public String name = null ; public Statement apply ( final Statement base , final Description description ) { return new Statement ( ) { @ Override public void evaluate ( ) throws Throwable { name = description . getMethodName ( ) ; base . evaluate ( ) ; } } ; } } public static class UsesCustomMethodRule { @ Rule public CustomTestName counter = new CustomTestName ( ) ; @ Test public void foo ( ) { assertEquals ( "foo" , counter . name ) ; } } @ Test public void useCustomMethodRule ( ) { assertThat ( testResult ( UsesCustomMethodRule . class ) , isSuccessful ( ) ) ; } public static class HasMethodReturningMethodRule { private MethodRule methodRule = new MethodRule ( ) { public Statement apply ( final Statement base , FrameworkMethod method , Object target ) { return new Statement ( ) { @ Override public void evaluate ( ) throws Throwable { ruleWasEvaluated = true ; base . evaluate ( ) ; } } ; } } ; @ Rule public MethodRule methodRule ( ) { return methodRule ; } @ Test public void doNothing ( ) { } } @ Test public void runsMethodRuleThatIsReturnedByMethod ( ) { ruleWasEvaluated = false ; assertThat ( testResult ( HasMethodReturningMethodRule . class ) , isSuccessful ( ) ) ; assertTrue ( ruleWasEvaluated ) ; } public static class HasMultipleMethodsReturningMethodRule { @ Rule public Increment methodRule1 ( ) { return new Increment ( ) ; } @ Rule public Increment methodRule2 ( ) { return new Increment ( ) ; } @ Test public void doNothing ( ) { } } @ Test public void runsAllMethodRulesThatAreReturnedByMethods ( ) { runCount = 0 ; assertThat ( testResult ( HasMultipleMethodsReturningMethodRule . class ) , isSuccessful ( ) ) ; assertEquals ( 2 , runCount ) ; } public static class CallsMethodReturningRuleOnlyOnce { int callCount = 0 ; private static class Dummy implements MethodRule { public Statement apply ( final Statement base , FrameworkMethod method , Object target ) { return new Statement ( ) { @ Override public void evaluate ( ) throws Throwable { base . evaluate ( ) ; } } ; } } @ Rule public MethodRule methodRule ( ) { callCount++ ; return new Dummy ( ) ; } @ Test public void doNothing ( ) { assertEquals ( 1 , callCount ) ; } } @ Test public void callsMethodReturningRuleOnlyOnce ( ) { assertThat ( testResult ( CallsMethodReturningRuleOnlyOnce . class ) , isSuccessful ( ) ) ; } }