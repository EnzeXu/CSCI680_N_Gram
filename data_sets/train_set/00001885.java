public class LocalGroupByGate extends MemorySpliceGate { private static final Logger LOG = LoggerFactory.getLogger( LocalGroupByGate.class ); private ListMultimap<Tuple, Tuple> valueMap; public LocalGroupByGate( FlowProcess flowProcess, Splice splice ) { super( flowProcess, splice ); } @Override protected boolean isBlockingStreamed() { return true; } private ListMultimap<Tuple, Tuple> initNewValueMap() { return Multimaps.synchronizedListMultimap( ArrayListMultimap.<Tuple, Tuple>create() ); } @Override public void prepare() { super.prepare(); valueMap = initNewValueMap(); } @Override public void start( Duct previous ) { } @Override public void receive( Duct previous, int ordinal, TupleEntry incomingEntry ) { Tuple valuesTuple = incomingEntry.getTupleCopy(); Tuple groupTuple = keyBuilder[ 0 ].makeResult( valuesTuple, null ); groupTuple = getDelegatedTuple( groupTuple ); keys.add( groupTuple ); valueMap.put( groupTuple, valuesTuple ); } @Override public void complete( Duct previous ) { if( count.decrementAndGet() != 0 ) return; next.start( this ); Iterator<Tuple> iterator = keys.iterator(); while( iterator.hasNext() ) { Tuple groupTuple = iterator.next(); iterator.remove(); keyEntry.setTuple( groupTuple ); List<Tuple> tuples = valueMap.get( groupTuple ); if( valueComparators != null ) Collections.sort( tuples, valueComparators[ 0 ] ); tupleEntryIterator.reset( tuples.iterator() ); try { next.receive( this, 0, grouping ); } catch( StopDataNotificationException exception ) { LOG.info( "received stop data notification: {}", exception.getMessage() ); break; } tuples.clear(); } keys = createKeySet(); valueMap = initNewValueMap(); count.set( numIncomingEventingPaths ); next.complete( this ); } }