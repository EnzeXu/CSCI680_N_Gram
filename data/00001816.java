public class JSONCopyAsFunctionTest extends CascadingTestCase { @ Test public void testCopyAs ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , JSONCoercibleType . TYPE ) , Tuple . size ( 1 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; CopySpec copySpec = new CopySpec ( ) . from ( "/person" ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertEquals ( "John Doe" , ( ( ObjectNode ) value ) . get ( "name" ) . textValue ( ) ) ; assertEquals ( 50 , ( ( ObjectNode ) value ) . get ( "age" ) . intValue ( ) ) ; assertEquals ( "123-45-6789" , ( ( ObjectNode ) value ) . get ( "ssn" ) . textValue ( ) ) ; } @ Test public void testCopyAsPredicate ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , JSONCoercibleType . TYPE ) , Tuple . size ( 1 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; CopySpec copySpec = new CopySpec ( ) . from ( "/person" , new JSONStringPointerFilter ( "/name" , "John Doe" ) ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertEquals ( "John Doe" , ( ( ObjectNode ) value ) . get ( "name" ) . textValue ( ) ) ; assertEquals ( 50 , ( ( ObjectNode ) value ) . get ( "age" ) . intValue ( ) ) ; assertEquals ( "123-45-6789" , ( ( ObjectNode ) value ) . get ( "ssn" ) . textValue ( ) ) ; } @ Test public void testCopyAsPredicateNegate ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , JSONCoercibleType . TYPE ) , Tuple . size ( 1 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; CopySpec copySpec = new CopySpec ( ) . from ( "/person" , new JSONStringPointerFilter ( "/name" , "John Doe" ) . negate ( ) ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertNull ( ( ( ObjectNode ) value ) . get ( "name" ) ) ; assertNull ( ( ( ObjectNode ) value ) . get ( "age" ) ) ; assertNull ( ( ( ObjectNode ) value ) . get ( "ssn" ) ) ; } @ Test public void testCopyAsPredicateBoolean ( ) throws Exception { Fields fields = new Fields ( "json" , JSONCoercibleType . TYPE ) . append ( new Fields ( "result" , JSONCoercibleType . TYPE ) ) ; TupleEntry entry = new TupleEntry ( fields , Tuple . size ( 2 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; entry . setObject ( 1 , JSONData . simple ) ; CopySpec copySpec = new CopySpec ( ) . from ( "/person" , new JSONBooleanPointerFilter ( "/human" , true ) ) ; JSONCopyIntoFunction function = new JSONCopyIntoFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertEquals ( "value" , ( ( ObjectNode ) value ) . get ( "existing" ) . textValue ( ) ) ; assertEquals ( "John Doe" , ( ( ObjectNode ) value ) . get ( "name" ) . textValue ( ) ) ; assertEquals ( true , ( ( ObjectNode ) value ) . get ( "human" ) . booleanValue ( ) ) ; assertEquals ( 50 , ( ( ObjectNode ) value ) . get ( "age" ) . intValue ( ) ) ; assertEquals ( "123-45-6789" , ( ( ObjectNode ) value ) . get ( "ssn" ) . textValue ( ) ) ; } @ Test public void testCopyAsPredicateBooleanNegate ( ) throws Exception { Fields fields = new Fields ( "json" , JSONCoercibleType . TYPE ) . append ( new Fields ( "result" , JSONCoercibleType . TYPE ) ) ; TupleEntry entry = new TupleEntry ( fields , Tuple . size ( 2 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; entry . setObject ( 1 , JSONData . simple ) ; CopySpec copySpec = new CopySpec ( ) . from ( "/person" , new JSONBooleanPointerFilter ( "/human" , true ) . negate ( ) ) ; JSONCopyIntoFunction function = new JSONCopyIntoFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertEquals ( "value" , ( ( ObjectNode ) value ) . get ( "existing" ) . textValue ( ) ) ; assertNull ( ( ( ObjectNode ) value ) . get ( "name" ) ) ; assertNull ( ( ( ObjectNode ) value ) . get ( "human" ) ) ; assertNull ( ( ( ObjectNode ) value ) . get ( "age" ) ) ; assertNull ( ( ( ObjectNode ) value ) . get ( "ssn" ) ) ; } @ Test public void testCopyAsArrayPredicate ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , JSONCoercibleType . TYPE ) , Tuple . size ( 1 ) ) ; entry . setObject ( 0 , JSONData . people ) ; CopySpec copySpec = new CopySpec ( ) . from ( "/peopleage" ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertEquals ( "John" , ( ( ObjectNode ) value ) . findPath ( "firstName" ) . textValue ( ) ) ; assertEquals ( 50 , ( ( ObjectNode ) value ) . findPath ( "age" ) . intValue ( ) ) ; assertEquals ( null , ( ( ObjectNode ) value ) . findValue ( "ssn" ) ) ; } @ Test public void testCopyIncludeDescent ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , JSONCoercibleType . TYPE ) , Tuple . size ( 1 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; CopySpec copySpec = new CopySpec ( ) . include ( "/person/firstName" ) . include ( "age" ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertEquals ( "John" , ( ( ObjectNode ) value ) . findPath ( "firstName" ) . textValue ( ) ) ; assertEquals ( 50 , ( ( ObjectNode ) value ) . findPath ( "age" ) . intValue ( ) ) ; assertEquals ( null , ( ( ObjectNode ) value ) . findValue ( "ssn" ) ) ; } @ Test public void testCopyIncludeFrom2 ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , JSONCoercibleType . TYPE ) , Tuple . size ( 1 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; CopySpec copySpec = new CopySpec ( ) . fromInclude ( "/person" , "/firstName" , "/age" ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertEquals ( "John" , ( ( ObjectNode ) value ) . get ( "firstName" ) . textValue ( ) ) ; assertEquals ( 50 , ( ( ObjectNode ) value ) . get ( "age" ) . intValue ( ) ) ; assertEquals ( null , ( ( ObjectNode ) value ) . get ( "ssn" ) ) ; } @ Test public void testCopyExcludeFrom ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , JSONCoercibleType . TYPE ) , Tuple . size ( 1 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; CopySpec copySpec = new CopySpec ( ) . fromExclude ( "/person" , "/ssn" , "/children" ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertEquals ( "John Doe" , ( ( ObjectNode ) value ) . get ( "name" ) . textValue ( ) ) ; assertEquals ( 50 , ( ( ObjectNode ) value ) . get ( "age" ) . intValue ( ) ) ; assertEquals ( null , ( ( ObjectNode ) value ) . get ( "ssn" ) ) ; assertEquals ( null , ( ( ObjectNode ) value ) . get ( "children" ) ) ; } @ Test public void testCopyExclude ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , JSONCoercibleType . TYPE ) , Tuple . size ( 1 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; CopySpec copySpec = new CopySpec ( ) . exclude ( "/person/ssn" , "/person/children" ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertEquals ( "John Doe" , ( ( ObjectNode ) value ) . findPath ( "name" ) . textValue ( ) ) ; assertEquals ( 50 , ( ( ObjectNode ) value ) . findPath ( "age" ) . intValue ( ) ) ; assertEquals ( null , ( ( ObjectNode ) value ) . findValue ( "ssn" ) ) ; assertEquals ( null , ( ( ObjectNode ) value ) . findValue ( "children" ) ) ; } @ Test public void testCopyExcludeDescent ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , JSONCoercibleType . TYPE ) , Tuple . size ( 1 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; CopySpec copySpec = new CopySpec ( ) . exclude ( "value" ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertNotNull ( ( ( ObjectNode ) value ) . get ( "person" ) ) ; value = ( ( ObjectNode ) value ) . get ( "person" ) ; assertNull ( ( ( ObjectNode ) value ) . get ( "measure" ) . get ( "value" ) ) ; } @ Test public void testCoerce ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , JSONCoercibleType . TYPE ) , Tuple . size ( 1 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; CopySpec copySpec = new CopySpec ( ) . fromTransform ( "/person/measure" , "/value" , JSONPrimitiveTransforms . TO_FLOAT ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertEquals ( JsonNodeType . NUMBER , ( ( ObjectNode ) value ) . get ( "value" ) . getNodeType ( ) ) ; assertEquals ( FloatNode . class , ( ( ObjectNode ) value ) . get ( "value" ) . getClass ( ) ) ; assertEquals ( 100 . 0F , ( ( ObjectNode ) value ) . get ( "value" ) . floatValue ( ) ) ; } @ Test public void testCoerceZero ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , JSONCoercibleType . TYPE ) , Tuple . size ( 1 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; CopySpec copySpec = new CopySpec ( ) . fromTransform ( "/person/zero" , "/zeroValue" , JSONPrimitiveTransforms . TO_FLOAT ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertEquals ( JsonNodeType . NUMBER , ( ( ObjectNode ) value ) . get ( "zeroValue" ) . getNodeType ( ) ) ; assertEquals ( FloatNode . class , ( ( ObjectNode ) value ) . get ( "zeroValue" ) . getClass ( ) ) ; assertEquals ( 0 . 0F , ( ( ObjectNode ) value ) . get ( "zeroValue" ) . floatValue ( ) ) ; assertEquals ( "0 . 0" , ( ( ObjectNode ) value ) . get ( "zeroValue" ) . asText ( ) ) ; } @ Test public void testCoerceArray ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , JSONCoercibleType . TYPE ) , Tuple . size ( 1 ) ) ; entry . setObject ( 0 , JSONData . nested ) ; CopySpec copySpec = new CopySpec ( ) . fromTransform ( "/person" , "/measures/*/value" , JSONPrimitiveTransforms . TO_FLOAT ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; Object value = result . iterator ( ) . next ( ) . getObject ( 0 ) ; assertNotNull ( value ) ; assertEquals ( JsonNodeType . ARRAY , ( ( ObjectNode ) value ) . get ( "measures" ) . getNodeType ( ) ) ; assertEquals ( FloatNode . class , ( ( ObjectNode ) value ) . get ( "measures" ) . get ( 0 ) . get ( "value" ) . getClass ( ) ) ; assertEquals ( 1000 . 0F , ( ( ObjectNode ) value ) . get ( "measures" ) . get ( 0 ) . get ( "value" ) . floatValue ( ) ) ; assertEquals ( FloatNode . class , ( ( ObjectNode ) value ) . get ( "measures" ) . get ( 1 ) . get ( "value" ) . getClass ( ) ) ; assertEquals ( 2000 . 0F , ( ( ObjectNode ) value ) . get ( "measures" ) . get ( 1 ) . get ( "value" ) . floatValue ( ) ) ; } @ Test public void testResettableTransform ( ) throws Exception { TupleEntry entry = new TupleEntry ( new Fields ( "json" , "set-text" ) . applyTypes ( JSONCoercibleType . TYPE , String . class ) , Tuple . size ( 2 ) ) ; CopySpec copySpec = new CopySpec ( ) . fromTransform ( "/person" , "/name" , new JSONSetTextTransform ( "set-text" , "value1" ) ) ; JSONCopyAsFunction function = new JSONCopyAsFunction ( new Fields ( "result" ) , copySpec ) ; entry . setObject ( 0 , JSONData . nested ) ; entry . setObject ( 1 , "value2" ) ; TupleListCollector result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; assertNotNull ( result . iterator ( ) . next ( ) . getObject ( 0 ) ) ; assertEquals ( "value2" , ( ( ObjectNode ) result . iterator ( ) . next ( ) . getObject ( 0 ) ) . get ( "name" ) . textValue ( ) ) ; entry . setObject ( 0 , JSONData . nested ) ; entry . setObject ( 1 , "value3" ) ; result = invokeFunction ( function , entry , new Fields ( "result" ) ) ; assertNotNull ( result . iterator ( ) . next ( ) . getObject ( 0 ) ) ; assertEquals ( "value3" , ( ( ObjectNode ) result . iterator ( ) . next ( ) . getObject ( 0 ) ) . get ( "name" ) . textValue ( ) ) ; } }