public class ValueAssertionsTest extends CascadingTestCase { public ValueAssertionsTest ( ) { } private TupleEntry getEntry ( Tuple tuple ) { return new TupleEntry ( Fields . size ( tuple . size ( ) ) , tuple ) ; } private void assertFail ( ValueAssertion assertion , TupleEntry tupleEntry ) { ConcreteCall concreteCall = getOperationCall ( tupleEntry ) ; assertion . prepare ( FlowProcess . NULL , concreteCall ) ; try { assertion . doAssert ( FlowProcess . NULL , concreteCall ) ; fail ( ) ; } catch ( AssertionException exception ) { } } private ConcreteCall getOperationCall ( TupleEntry tupleEntry ) { ConcreteCall operationCall = new ConcreteCall ( tupleEntry . getFields ( ) ) ; operationCall . setArguments ( tupleEntry ) ; return operationCall ; } private void assertPass ( ValueAssertion assertion , TupleEntry tupleEntry ) { ConcreteCall concreteCall = getOperationCall ( tupleEntry ) ; assertion . prepare ( FlowProcess . NULL , concreteCall ) ; assertion . doAssert ( FlowProcess . NULL , concreteCall ) ; } @ Test public void testNotNull ( ) { ValueAssertion assertion = new AssertNotNull ( ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; } @ Test public void testNull ( ) { ValueAssertion assertion = new AssertNull ( ) ; assertFail ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( null , null ) ) ) ; } @ Test public void testEquals ( ) { ValueAssertion assertion = new AssertEquals ( 1 ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( 1 , 1 , 1 , 1 , 1 , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; assertion = new AssertEquals ( "one" , "two" ) ; assertPass ( assertion , getEntry ( new Tuple ( "one" , "two" ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( null , null ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; } @ Test public void testNotEquals ( ) { ValueAssertion assertion = new AssertNotEquals ( 4 ) ; assertFail ( assertion , getEntry ( new Tuple ( 4 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 , 1 , 1 , 1 , 1 , 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; assertion = new AssertNotEquals ( "one1" , "two1" ) ; assertFail ( assertion , getEntry ( new Tuple ( "one1" , "two1" ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "one" , "two" ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( null , null ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; } @ Test public void testEqualsAll ( ) { ValueAssertion assertion = new AssertEqualsAll ( 1 ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 , 1 , 1 , 1 , 1 , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; } @ Test public void testMatches ( ) { ValueAssertion assertion = new AssertMatches ( "^1$" ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "1" ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( 1 , 1 , 1 , 1 , 1 , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; assertion = new AssertMatches ( "^1$" , false ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "1" ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( 1 , 1 , 1 , 1 , 1 , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; assertion = new AssertMatches ( "^1$" , true ) ; assertFail ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "1" ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 , 1 , 1 , 1 , 1 , 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; } @ Test public void testMatchesAll ( ) { ValueAssertion assertion = new AssertMatchesAll ( "^1$" ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "1" ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 , 1 , 1 , 1 , 1 , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( 1 , 1 , 1 , 0 , 1 , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; assertion = new AssertMatchesAll ( "^1$" , false ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "1" ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 , 1 , 1 , 1 , 1 , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( 1 , 1 , 1 , 0 , 1 , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; assertion = new AssertMatchesAll ( "^1$" , true ) ; assertFail ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "1" ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( 1 , 1 , 1 , 1 , 1 , 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; } @ Test public void testTupleEquals ( ) { ValueAssertion assertion = new AssertSizeEquals ( 1 ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; } @ Test public void testTupleLessThan ( ) { ValueAssertion assertion = new AssertSizeLessThan ( 2 ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; } @ Test public void testTupleMoreThan ( ) { ValueAssertion assertion = new AssertSizeMoreThan ( 1 ) ; assertFail ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "0" , 1 ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; } }