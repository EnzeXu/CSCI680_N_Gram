public class SparseTupleFieldsComparatorTest extends PlatformTestCase { public SparseTupleFieldsComparatorTest() { } @Test public void testCompare() { List<Tuple> result = new ArrayList<Tuple>(); result.add( new Tuple( "1", "1", "1" ) ); result.add( new Tuple( "2", "10", "1" ) ); result.add( new Tuple( "3", "1", "1" ) ); runTest( new Fields( "a", "b" ), result, null ); result = new ArrayList<Tuple>(); result.add( new Tuple( "1", "1", "1" ) ); result.add( new Tuple( "3", "1", "1" ) ); result.add( new Tuple( "2", "10", "1" ) ); runTest( new Fields( "b", "a" ), result, null ); result = new ArrayList<Tuple>(); result.add( new Tuple( "2", "10", "1" ) ); result.add( new Tuple( "1", "1", "1" ) ); result.add( new Tuple( "3", "1", "1" ) ); runTest( new Fields( "c" ), result, getPlatform().getStringComparator( true ) ); } private void runTest( Fields sortFields, List<Tuple> result, Comparator defaultComparator ) { Fields fields = new Fields( "a", "b", "c" ); List<Tuple> list = new ArrayList<Tuple>(); list.add( new Tuple( "2", "10", "1" ) ); list.add( new Tuple( "1", "1", "1" ) ); list.add( new Tuple( "3", "1", "1" ) ); Collections.sort( list, new SparseTupleComparator( fields, sortFields, defaultComparator ) ); assertEquals( result, list ); } }