public class ArrayComparisonFailureTest { private static final String ARRAY_COMPARISON_FAILURE_411 = "arrayComparisonFailure_411" ; private static final String ARRAY_COMPARISON_FAILURE_412 = "arrayComparisonFailure_412" ; @ Test public void classShouldAccept411Version ( ) throws Exception { assertFailureSerializableFromOthers ( ARRAY_COMPARISON_FAILURE_411 ) ; } @ Test public void classShouldAccept412Version ( ) throws Exception { assertFailureSerializableFromOthers ( ARRAY_COMPARISON_FAILURE_412 ) ; } private void assertFailureSerializableFromOthers ( String failureFileName ) throws IOException , ClassNotFoundException { try { assertArrayEquals ( new int [ ] { 0 , 1 } , new int [ ] { 0 , 5 } ) ; fail ( ) ; } catch ( ArrayComparisonFailure e ) { ArrayComparisonFailure arrayComparisonFailureFromFile = deserializeFailureFromFile ( failureFileName ) ; assertNotNull ( "ArrayComparisonFailure . getCause ( ) should fallback to the deprecated fCause field" + " for compatibility with older versions of junit4 that didn't use Throwable . initCause ( ) . " , arrayComparisonFailureFromFile . getCause ( ) ) ; assertEquals ( e . getCause ( ) . toString ( ) , arrayComparisonFailureFromFile . getCause ( ) . toString ( ) ) ; assertEquals ( e . toString ( ) , arrayComparisonFailureFromFile . toString ( ) ) ; } } private ArrayComparisonFailure deserializeFailureFromFile ( String fileName ) throws IOException , ClassNotFoundException { InputStream resource = getClass ( ) . getResourceAsStream ( fileName ) ; ObjectInputStream objectInputStream = new ObjectInputStream ( resource ) ; return ( ArrayComparisonFailure ) objectInputStream . readObject ( ) ; } }