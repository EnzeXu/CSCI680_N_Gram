public class AggregatorsTest extends CascadingTestCase { public AggregatorsTest() { } @Test public void testAverage() { Aggregator aggregator = new Average(); Tuple[] arguments = new Tuple[]{new Tuple( new Double( 1.0 ) ), new Tuple( new Double( 3.0 ) ), new Tuple( new Double( 2.0 ) ), new Tuple( new Double( 4.0 ) ), new Tuple( new Double( -5.0 ) )}; Fields resultFields = new Fields( "field" ); TupleListCollector resultEntryCollector = invokeAggregator( aggregator, arguments, resultFields ); Tuple tuple = resultEntryCollector.iterator().next(); assertEquals( "got expected value after aggregate", 1.0, tuple.getDouble( 0 ), 0.0d ); } @Test public void testCount() { Aggregator aggregator = new Count(); Tuple[] arguments = new Tuple[]{new Tuple( new Double( 1.0 ) ), new Tuple( new Double( 3.0 ) ), new Tuple( new Double( 2.0 ) ), new Tuple( new Double( 4.0 ) ), new Tuple( new Double( -5.0 ) )}; Fields resultFields = new Fields( "field" ); TupleListCollector resultEntryCollector = invokeAggregator( aggregator, arguments, resultFields ); Tuple tuple = resultEntryCollector.iterator().next(); assertEquals( "got expected value after aggregate", 5, tuple.getInteger( 0 ) ); } @Test public void testFirst() { Aggregator aggregator = new First(); Tuple[] arguments = new Tuple[]{new Tuple( new Double( 1.0 ) ), new Tuple( new Double( 3.0 ) ), new Tuple( new Double( 2.0 ) ), new Tuple( new Double( 4.0 ) ), new Tuple( new Double( -5.0 ) )}; Fields resultFields = new Fields( "field" ); TupleListCollector resultEntryCollector = invokeAggregator( aggregator, arguments, resultFields ); Tuple tuple = resultEntryCollector.iterator().next(); assertEquals( "got expected value after aggregate", 1.0, tuple.getDouble( 0 ), 0.0d ); } @Test public void testFirstN() { Aggregator aggregator = new First( 3 ); Tuple[] arguments = new Tuple[]{new Tuple( new Double( 1.0 ) ), new Tuple( new Double( 3.0 ) ), new Tuple( new Double( 2.0 ) ), new Tuple( new Double( 4.0 ) ), new Tuple( new Double( -5.0 ) )}; Fields resultFields = new Fields( "field" ); TupleListCollector resultEntryCollector = invokeAggregator( aggregator, arguments, resultFields ); Iterator<Tuple> iterator = resultEntryCollector.iterator(); assertEquals( "got expected value after aggregate", 1.0, iterator.next().getDouble( 0 ), 0.0d ); assertEquals( "got expected value after aggregate", 3.0, iterator.next().getDouble( 0 ), 0.0d ); assertEquals( "got expected value after aggregate", 2.0, iterator.next().getDouble( 0 ), 0.0d ); } @Test public void testLast() { Aggregator aggregator = new Last(); Tuple[] arguments = new Tuple[]{new Tuple( new Double( 1.0 ) ), new Tuple( new Double( 3.0 ) ), new Tuple( new Double( 2.0 ) ), new Tuple( new Double( 4.0 ) ), new Tuple( new Double( -5.0 ) )}; Fields resultFields = new Fields( "field" ); TupleListCollector resultEntryCollector = invokeAggregator( aggregator, arguments, resultFields ); Tuple tuple = resultEntryCollector.iterator().next(); assertEquals( "got expected value after aggregate", -5.0, tuple.getDouble( 0 ), 0.0d ); } @Test public void testSum() { Aggregator aggregator = new Sum(); Tuple[] arguments = new Tuple[]{new Tuple( new Double( 1.0 ) ), new Tuple( new Double( 3.0 ) ), new Tuple( new Double( 2.0 ) ), new Tuple( new Double( 4.0 ) ), new Tuple( new Double( -5.0 ) )}; Fields resultFields = new Fields( "field" ); TupleListCollector resultEntryCollector = invokeAggregator( aggregator, arguments, resultFields ); Tuple tuple = resultEntryCollector.iterator().next(); assertEquals( "got expected value after aggregate", 5.0, tuple.getDouble( 0 ), 0.0d ); } @Test public void testMaxValue() { Aggregator aggregator = new MaxValue(); Tuple[] arguments = new Tuple[]{new Tuple( new Double( 1.0 ) ), new Tuple( new Double( 3.0 ) ), new Tuple( new Double( 2.0 ) ), new Tuple( new Double( 4.0 ) ), new Tuple( new Double( -5.0 ) )}; Fields resultFields = new Fields( "field" ); TupleListCollector resultEntryCollector = invokeAggregator( aggregator, arguments, resultFields ); Tuple tuple = resultEntryCollector.iterator().next(); assertEquals( "got expected value after aggregate", 4.0, tuple.getDouble( 0 ), 0.0d ); } @Test public void testMinValue() { Aggregator aggregator = new MinValue(); Tuple[] arguments = new Tuple[]{new Tuple( new Double( 1.0 ) ), new Tuple( new Double( 3.0 ) ), new Tuple( new Double( 2.0 ) ), new Tuple( new Double( 4.0 ) ), new Tuple( new Double( -5.0 ) )}; Fields resultFields = new Fields( "field" ); TupleListCollector resultEntryCollector = invokeAggregator( aggregator, arguments, resultFields ); Tuple tuple = resultEntryCollector.iterator().next(); assertEquals( "got expected value after aggregate", -5.0, tuple.getDouble( 0 ), 0.0d ); } @Test public void testMaxValueNonNumber() { Aggregator aggregator = new MaxValue(); Tuple[] arguments = new Tuple[]{new Tuple( 'a' ), new Tuple( 'b' ), new Tuple( 'c' ), new Tuple( 'd' ), new Tuple( 'e' )}; Fields resultFields = new Fields( "field" ); TupleListCollector resultEntryCollector = invokeAggregator( aggregator, arguments, resultFields ); Tuple tuple = resultEntryCollector.iterator().next(); assertEquals( "got expected value after aggregate", 'e', tuple.getChar( 0 ) ); } @Test public void testMinValueNonNumber() { Aggregator aggregator = new MinValue(); Tuple[] arguments = new Tuple[]{new Tuple( 'a' ), new Tuple( 'b' ), new Tuple( 'c' ), new Tuple( 'd' ), new Tuple( 'e' )}; Fields resultFields = new Fields( "field" ); TupleListCollector resultEntryCollector = invokeAggregator( aggregator, arguments, resultFields ); Tuple tuple = resultEntryCollector.iterator().next(); assertEquals( "got expected value after aggregate", 'a', tuple.getChar( 0 ) ); } }