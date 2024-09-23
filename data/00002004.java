public class JSONGetAllFunctionTest extends CascadingTestCase { @Test public void testGetAll() { TupleEntry entry = new TupleEntry( new Fields( "json", JSONCoercibleType.TYPE ), Tuple.size( 1 ) ); entry.setObject( 0, JSONData.people ); JSONGetAllFunction function = new JSONGetAllFunction( "/peoplename", new Fields( "result", String.class ), "" ); TupleListCollector result = invokeFunction( function, entry, new Fields( "result" ) ); assertEquals( 2, result.size() ); Iterator<Tuple> iterator = result.iterator(); Object value = iterator.next().getObject( 0 ); assertNotNull( value ); assertEquals( "John Doe", value ); value = iterator.next().getObject( 0 ); assertNotNull( value ); assertEquals( "Jane Doe", value ); } }