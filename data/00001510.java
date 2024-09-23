public class ValueAssertionsTest extends CascadingTestCase { public ValueAssertionsTest ( ) { } private TupleEntry getEntry ( Tuple tuple ) { return new TupleEntry ( Fields . size ( tuple . size ( ) ) , tuple ) ; } private ConcreteCall getOperationCall ( TupleEntry tupleEntry ) { ConcreteCall operationCall = new ConcreteCall ( tupleEntry . getFields ( ) ) ; operationCall . setArguments ( tupleEntry ) ; return operationCall ; } private void assertFail ( ValueAssertion assertion , TupleEntry tupleEntry ) { ConcreteCall concreteCall = getOperationCall ( tupleEntry ) ; assertion . prepare ( FlowProcess . NULL , concreteCall ) ; try { assertion . doAssert ( FlowProcess . NULL , concreteCall ) ; fail ( ) ; } catch ( AssertionException exception ) { } } private void assertPass ( ValueAssertion assertion , TupleEntry tupleEntry ) { ConcreteCall concreteCall = getOperationCall ( tupleEntry ) ; assertion . prepare ( FlowProcess . NULL , concreteCall ) ; assertion . doAssert ( FlowProcess . NULL , concreteCall ) ; } @ Test public void testExpression ( ) { ValueAssertion assertion = new AssertExpression ( "$0 == 1" , int . class ) ; assertPass ( assertion , getEntry ( new Tuple ( 1 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( ( Comparable ) null ) ) ) ; assertPass ( assertion , getEntry ( new Tuple ( "1" , 0 ) ) ) ; assertFail ( assertion , getEntry ( new Tuple ( "0" , null ) ) ) ; } }