public class Unique extends SubAssembly { public enum Include { ALL, NO_NULLS } public enum Cache { Num_Keys_Flushed, Num_Keys_Hit, Num_Keys_Missed } public static class FilterPartialDuplicates extends BaseOperation<CascadingCache<Tuple, Object>> implements Filter<CascadingCache<Tuple, Object>> { private final static Object NULL_VALUE = new Object(); private int capacity = 0; private Include include = Include.ALL; private TupleHasher tupleHasher; public FilterPartialDuplicates() { } @ConstructorProperties({"capacity"}) public FilterPartialDuplicates( int capacity ) { this.capacity = capacity; } @ConstructorProperties({"include", "capacity"}) public FilterPartialDuplicates( Include include, int capacity ) { this( include, capacity, null ); } @ConstructorProperties({"include", "capacity", "tupleHasher"}) public FilterPartialDuplicates( Include include, int capacity, TupleHasher tupleHasher ) { this.capacity = capacity; this.include = include == null ? this.include : include; this.tupleHasher = tupleHasher; } @Override public void prepare( final FlowProcess flowProcess, OperationCall<CascadingCache<Tuple, Object>> operationCall ) { CacheEvictionCallback callback = new CacheEvictionCallback() { @Override public void evict( Map.Entry entry ) { flowProcess.increment( Cache.Num_Keys_Flushed, 1 ); } }; FactoryLoader loader = FactoryLoader.getInstance(); BaseCacheFactory cacheFactory = loader.loadFactoryFrom( flowProcess, UniqueProps.UNIQUE_CACHE_FACTORY, UniqueProps.DEFAULT_CACHE_FACTORY_CLASS ); if( cacheFactory == null ) throw new CascadingException( "unable to load cache factory, please check your '" + UniqueProps.UNIQUE_CACHE_FACTORY + "' setting." ); CascadingCache cache = cacheFactory.create( flowProcess ); cache.setCacheEvictionCallback( callback ); Integer cacheCapacity = capacity; if( capacity == 0 ) { cacheCapacity = flowProcess.getIntegerProperty( UniqueProps.UNIQUE_CACHE_CAPACITY ); if( cacheCapacity == null ) cacheCapacity = UniqueProps.UNIQUE_DEFAULT_CAPACITY; } cache.setCapacity( cacheCapacity.intValue() ); cache.initialize(); operationCall.setContext( cache ); } @Override public boolean isRemove( FlowProcess flowProcess, FilterCall<CascadingCache<Tuple, Object>> filterCall ) { Tuple args = TupleHasher.wrapTuple( tupleHasher, filterCall.getArguments().getTuple() ); switch( include ) { case ALL: break; case NO_NULLS: if( Tuples.frequency( args, null ) == args.size() ) return true; break; } if( filterCall.getContext().containsKey( args ) ) { flowProcess.increment( Cache.Num_Keys_Hit, 1 ); return true; } filterCall.getContext().put( TupleHasher.wrapTuple( tupleHasher, filterCall.getArguments().getTupleCopy() ), NULL_VALUE ); flowProcess.increment( Cache.Num_Keys_Missed, 1 ); return false; } @Override public void cleanup( FlowProcess flowProcess, OperationCall<CascadingCache<Tuple, Object>> operationCall ) { operationCall.setContext( null ); } @Override public boolean equals( Object object ) { if( this == object ) return true; if( !( object instanceof FilterPartialDuplicates ) ) return false; if( !super.equals( object ) ) return false; FilterPartialDuplicates that = (FilterPartialDuplicates) object; if( capacity != that.capacity ) return false; return true; } @Override public int hashCode() { int result = super.hashCode(); result = 31 * result + capacity; return result; } } @ConstructorProperties({"pipe", "uniqueFields"}) public Unique( Pipe pipe, Fields uniqueFields ) { this( null, pipe, uniqueFields ); } @ConstructorProperties({"pipe", "uniqueFields", "include"}) public Unique( Pipe pipe, Fields uniqueFields, Include include ) { this( null, pipe, uniqueFields, include ); } @ConstructorProperties({"pipe", "uniqueFields", "capacity"}) public Unique( Pipe pipe, Fields uniqueFields, int capacity ) { this( null, pipe, uniqueFields, capacity ); } @ConstructorProperties({"pipe", "uniqueFields", "include", "capacity"}) public Unique( Pipe pipe, Fields uniqueFields, Include include, int capacity ) { this( null, pipe, uniqueFields, include, capacity ); } @ConstructorProperties({"name", "pipe", "uniqueFields"}) public Unique( String name, Pipe pipe, Fields uniqueFields ) { this( name, pipe, uniqueFields, null ); } @ConstructorProperties({"name", "pipe", "uniqueFields", "include"}) public Unique( String name, Pipe pipe, Fields uniqueFields, Include include ) { this( name, pipe, uniqueFields, include, 0 ); } @ConstructorProperties({"name", "pipe", "uniqueFields", "capacity"}) public Unique( String name, Pipe pipe, Fields uniqueFields, int capacity ) { this( name, Pipe.pipes( pipe ), uniqueFields, capacity ); } @ConstructorProperties({"name", "pipe", "uniqueFields", "include", "capacity"}) public Unique( String name, Pipe pipe, Fields uniqueFields, Include include, int capacity ) { this( name, Pipe.pipes( pipe ), uniqueFields, include, capacity ); } @ConstructorProperties({"pipes", "uniqueFields"}) public Unique( Pipe[] pipes, Fields uniqueFields ) { this( null, pipes, uniqueFields ); } @ConstructorProperties({"pipes", "uniqueFields", "include"}) public Unique( Pipe[] pipes, Fields uniqueFields, Include include ) { this( null, pipes, uniqueFields, include ); } @ConstructorProperties({"pipes", "uniqueFields", "capacity"}) public Unique( Pipe[] pipes, Fields uniqueFields, int capacity ) { this( null, pipes, uniqueFields, capacity ); } @ConstructorProperties({"pipes", "uniqueFields", "include", "capacity"}) public Unique( Pipe[] pipes, Fields uniqueFields, Include include, int capacity ) { this( null, pipes, uniqueFields, include, capacity ); } @ConstructorProperties({"name", "pipes", "uniqueFields"}) public Unique( String name, Pipe[] pipes, Fields uniqueFields ) { this( name, pipes, uniqueFields, null ); } @ConstructorProperties({"name", "pipes", "uniqueFields", "include"}) public Unique( String name, Pipe[] pipes, Fields uniqueFields, Include include ) { this( name, pipes, uniqueFields, include, 0 ); } @ConstructorProperties({"name", "pipes", "uniqueFields", "capacity"}) public Unique( String name, Pipe[] pipes, Fields uniqueFields, int capacity ) { this( name, pipes, uniqueFields, null, capacity ); } @ConstructorProperties({"name", "pipes", "uniqueFields", "include", "capacity"}) public Unique( String name, Pipe[] pipes, Fields uniqueFields, Include include, int capacity ) { super( pipes ); if( uniqueFields == null ) throw new IllegalArgumentException( "uniqueFields may not be null" ); Pipe[] filters = new Pipe[ pipes.length ]; TupleHasher tupleHasher = null; Comparator[] comparators = uniqueFields.getComparators(); if( !TupleHasher.isNull( comparators ) ) tupleHasher = new TupleHasher( null, comparators ); FilterPartialDuplicates partialDuplicates = new FilterPartialDuplicates( include, capacity, tupleHasher ); for( int i = 0; i < filters.length; i++ ) filters[ i ] = new Each( pipes[ i ], uniqueFields, partialDuplicates ); Pipe pipe = new GroupBy( name, filters, uniqueFields ); pipe = new Every( pipe, Fields.ALL, new FirstNBuffer(), Fields.RESULTS ); setTails( pipe ); } }