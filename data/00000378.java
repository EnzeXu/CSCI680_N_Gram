public class TestedOnSupplierTest { public void foo ( @ TestedOn ( ints = { 1 } ) int x ) { } @ Test public void descriptionStatesParameterName ( ) throws Exception { TestedOnSupplier supplier = new TestedOnSupplier ( ) ; List < PotentialAssignment > assignments = supplier . getValueSources ( signatureOfFoo ( ) ) ; assertThat ( assignments . get ( 0 ) . getDescription ( ) , is ( "\"1\" < from ints > " ) ) ; } private ParameterSignature signatureOfFoo ( ) throws NoSuchMethodException { Method method = getClass ( ) . getMethod ( "foo" , int . class ) ; return ParameterSignature . signatures ( method ) . get ( 0 ) ; } }