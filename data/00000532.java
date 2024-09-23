public class MaxCore { private static final String MALFORMED_JUNIT_3_TEST_CLASS_PREFIX = "malformed JUnit 3 test class : " ; @ Deprecated public static MaxCore forFolder ( String folderName ) { return storedLocally ( new File ( folderName ) ) ; } public static MaxCore storedLocally ( File storedResults ) { return new MaxCore ( storedResults ) ; } private final MaxHistory history ; private MaxCore ( File storedResults ) { history = MaxHistory . forFolder ( storedResults ) ; } public Result run ( Class < ? > testClass ) { return run ( Request . aClass ( testClass ) ) ; } public Result run ( Request request ) { return run ( request , new JUnitCore ( ) ) ; } public Result run ( Request request , JUnitCore core ) { core . addListener ( history . listener ( ) ) ; return core . run ( sortRequest ( request ) . getRunner ( ) ) ; } public Request sortRequest ( Request request ) { if ( request instanceof SortingRequest ) { return request ; } List < Description > leaves = findLeaves ( request ) ; Collections . sort ( leaves , history . testComparator ( ) ) ; return constructLeafRequest ( leaves ) ; } private Request constructLeafRequest ( List < Description > leaves ) { final List < Runner > runners = new ArrayList < Runner > ( ) ; for ( Description each : leaves ) { runners . add ( buildRunner ( each ) ) ; } return new Request ( ) { @ Override public Runner getRunner ( ) { try { return new Suite ( ( Class < ? > ) null , runners ) { } ; } catch ( InitializationError e ) { return new ErrorReportingRunner ( null , e ) ; } } } ; } private Runner buildRunner ( Description each ) { if ( each . toString ( ) . equals ( "TestSuite with 0 tests" ) ) { return Suite . emptySuite ( ) ; } if ( each . toString ( ) . startsWith ( MALFORMED_JUNIT_3_TEST_CLASS_PREFIX ) ) { return new JUnit38ClassRunner ( new TestSuite ( getMalformedTestClass ( each ) ) ) ; } Class < ? > type = each . getTestClass ( ) ; if ( type == null ) { throw new RuntimeException ( "Can't build a runner from description [ " + each + " ] " ) ; } String methodName = each . getMethodName ( ) ; if ( methodName == null ) { return Request . aClass ( type ) . getRunner ( ) ; } return Request . method ( type , methodName ) . getRunner ( ) ; } private Class < ? > getMalformedTestClass ( Description each ) { try { return Class . forName ( each . toString ( ) . replace ( MALFORMED_JUNIT_3_TEST_CLASS_PREFIX , "" ) ) ; } catch ( ClassNotFoundException e ) { return null ; } } public List < Description > sortedLeavesForTest ( Request request ) { return findLeaves ( sortRequest ( request ) ) ; } private List < Description > findLeaves ( Request request ) { List < Description > results = new ArrayList < Description > ( ) ; findLeaves ( null , request . getRunner ( ) . getDescription ( ) , results ) ; return results ; } private void findLeaves ( Description parent , Description description , List < Description > results ) { if ( description . getChildren ( ) . isEmpty ( ) ) { if ( description . toString ( ) . equals ( "warning ( junit . framework . TestSuite$1 ) " ) ) { results . add ( Description . createSuiteDescription ( MALFORMED_JUNIT_3_TEST_CLASS_PREFIX + parent ) ) ; } else { results . add ( description ) ; } } else { for ( Description each : description . getChildren ( ) ) { findLeaves ( description , each , results ) ; } } } }