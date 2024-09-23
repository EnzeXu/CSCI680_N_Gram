public class ParentRunnerFilteringTest { private static Filter notThisMethodName ( final String methodName ) { return new Filter ( ) { @ Override public boolean shouldRun ( Description description ) { return description . getMethodName ( ) == null || !description . getMethodName ( ) . equals ( methodName ) ; } @ Override public String describe ( ) { return "don't run method name : " + methodName ; } } ; } private static class CountingFilter extends Filter { private final Map < Description , Integer > countMap = new HashMap < Description , Integer > ( ) ; @ Override public boolean shouldRun ( Description description ) { Integer count = countMap . get ( description ) ; if ( count == null ) { countMap . put ( description , 1 ) ; } else { countMap . put ( description , count + 1 ) ; } return true ; } @ Override public String describe ( ) { return "filter counter" ; } public int getCount ( final Description desc ) { if ( !countMap . containsKey ( desc ) ) { throw new IllegalArgumentException ( "Looking for " + desc + " , but only contains : " + countMap . keySet ( ) ) ; } return countMap . get ( desc ) ; } } public static class ExampleTest { @ Test public void test1 ( ) throws Exception { } } @ RunWith ( Suite . class ) @ SuiteClasses ( { ExampleTest . class } ) public static class ExampleSuite { } @ Test public void testSuiteFiltering ( ) throws Exception { Runner runner = Request . aClass ( ExampleSuite . class ) . getRunner ( ) ; Filter filter = notThisMethodName ( "test1" ) ; try { filter . apply ( runner ) ; } catch ( NoTestsRemainException e ) { return ; } fail ( "Expected 'NoTestsRemainException' due to complete filtering" ) ; } public static class SuiteWithUnmodifiableChildList extends Suite { public SuiteWithUnmodifiableChildList ( Class < ? > klass , RunnerBuilder builder ) throws InitializationError { super ( klass , builder ) ; } @ Override protected List < Runner > getChildren ( ) { return Collections . unmodifiableList ( super . getChildren ( ) ) ; } } @ RunWith ( SuiteWithUnmodifiableChildList . class ) @ SuiteClasses ( { ExampleTest . class } ) public static class ExampleSuiteWithUnmodifiableChildList { } @ Test public void testSuiteFilteringWithUnmodifiableChildList ( ) throws Exception { Runner runner = Request . aClass ( ExampleSuiteWithUnmodifiableChildList . class ) . getRunner ( ) ; Filter filter = notThisMethodName ( "test1" ) ; try { filter . apply ( runner ) ; } catch ( NoTestsRemainException e ) { return ; } fail ( "Expected 'NoTestsRemainException' due to complete filtering" ) ; } @ Test public void testRunSuiteFiltering ( ) throws Exception { Request request = Request . aClass ( ExampleSuite . class ) ; Request requestFiltered = request . filterWith ( notThisMethodName ( "test1" ) ) ; assertThat ( testResult ( requestFiltered ) , hasSingleFailureContaining ( "don't run method name : test1" ) ) ; } @ Test public void testCountClassFiltering ( ) throws Exception { JUnitCore junitCore = new JUnitCore ( ) ; Request request = Request . aClass ( ExampleTest . class ) ; CountingFilter countingFilter = new CountingFilter ( ) ; Request requestFiltered = request . filterWith ( countingFilter ) ; Result result = junitCore . run ( requestFiltered ) ; assertEquals ( 1 , result . getRunCount ( ) ) ; assertEquals ( 0 , result . getFailureCount ( ) ) ; Description desc = createTestDescription ( ExampleTest . class , "test1" ) ; assertEquals ( 1 , countingFilter . getCount ( desc ) ) ; } @ Test public void testCountSuiteFiltering ( ) throws Exception { Class < ExampleSuite > suiteClazz = ExampleSuite . class ; Class < ExampleTest > clazz = ExampleTest . class ; JUnitCore junitCore = new JUnitCore ( ) ; Request request = Request . aClass ( suiteClazz ) ; CountingFilter countingFilter = new CountingFilter ( ) ; Request requestFiltered = request . filterWith ( countingFilter ) ; Result result = junitCore . run ( requestFiltered ) ; assertEquals ( 1 , result . getRunCount ( ) ) ; assertEquals ( 0 , result . getFailureCount ( ) ) ; Description suiteDesc = createSuiteDescription ( clazz ) ; assertEquals ( 1 , countingFilter . getCount ( suiteDesc ) ) ; Description desc = createTestDescription ( ExampleTest . class , "test1" ) ; assertEquals ( 1 , countingFilter . getCount ( desc ) ) ; } }