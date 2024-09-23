public class CompositeFunction extends BaseOperation<CompositeFunction.Context> implements Function<CompositeFunction.Context> { public static final String COMPOSITE_FUNCTION_CAPACITY = "cascading.function.composite.cache.capacity"; public static final Class<? extends BaseCacheFactory> DEFAULT_CACHE_FACTORY_CLASS = LRUHashMapCacheFactory.class; public static String COMPOSITE_FUNCTION_CACHE_FACTORY = "cascading.function.composite.cachefactory.classname"; public static int COMPOSITE_FUNCTION_DEFAULT_CAPACITY = BaseCacheFactory.DEFAULT_CAPACITY; public enum Cache { Num_Keys_Flushed, Num_Keys_Hit, Num_Keys_Missed } public interface CoFunction extends Serializable { Fields getDeclaredFields(); Tuple aggregate( FlowProcess flowProcess, TupleEntry args, Tuple context ); Tuple complete( FlowProcess flowProcess, Tuple context ); } public static class Context { CascadingCache<Tuple, Tuple[]> lru; TupleEntry[] arguments; Tuple result; } private final Fields groupingFields; private final Fields[] argumentFields; private final Fields[] functorFields; private final CoFunction[] coFunctions; private final TupleHasher tupleHasher; private int capacity = 0; public CompositeFunction( Fields groupingFields, Fields argumentFields, CoFunction coFunction, int capacity ) { this( groupingFields, Fields.fields( argumentFields ), new CoFunction[]{coFunction}, capacity ); } public CompositeFunction( Fields groupingFields, Fields[] argumentFields, CoFunction[] coFunctions, int capacity ) { super( getFields( groupingFields, coFunctions ) ); this.groupingFields = groupingFields; this.argumentFields = argumentFields; this.coFunctions = coFunctions; this.capacity = capacity; this.functorFields = new Fields[ coFunctions.length ]; for( int i = 0; i < coFunctions.length; i++ ) this.functorFields[ i ] = coFunctions[ i ].getDeclaredFields(); Comparator[] hashers = TupleHasher.merge( functorFields ); if( !TupleHasher.isNull( hashers ) ) this.tupleHasher = new TupleHasher( null, hashers ); else this.tupleHasher = null; } private static Fields getFields( Fields groupingFields, CoFunction[] coFunctions ) { Fields fields = groupingFields; for( CoFunction functor : coFunctions ) fields = fields.append( functor.getDeclaredFields() ); return fields; } @Override public void prepare( final FlowProcess flowProcess, final OperationCall<CompositeFunction.Context> operationCall ) { Fields[] fields = new Fields[ coFunctions.length + 1 ]; fields[ 0 ] = groupingFields; for( int i = 0; i < coFunctions.length; i++ ) fields[ i + 1 ] = coFunctions[ i ].getDeclaredFields(); final Context context = new Context(); context.arguments = new TupleEntry[ coFunctions.length ]; for( int i = 0; i < context.arguments.length; i++ ) { Fields resolvedArgumentFields = operationCall.getArgumentFields(); int[] pos; if( argumentFields[ i ].isAll() ) pos = resolvedArgumentFields.getPos(); else pos = resolvedArgumentFields.getPos( argumentFields[ i ] ); Tuple narrow = TupleViews.createNarrow( pos ); Fields currentFields; if( this.argumentFields[ i ].isSubstitution() ) currentFields = resolvedArgumentFields.select( this.argumentFields[ i ] ); else currentFields = Fields.asDeclaration( this.argumentFields[ i ] ); context.arguments[ i ] = new TupleEntry( currentFields, narrow ); } context.result = TupleViews.createComposite( fields ); class Eviction implements CacheEvictionCallback<Tuple, Tuple[]> { @Override public void evict( Map.Entry<Tuple, Tuple[]> entry ) { completeFunctors( flowProcess, ( (FunctionCall) operationCall ).getOutputCollector(), context.result, entry ); incrementNumKeysFlushed( flowProcess ); } } BaseCacheFactory<Tuple, Tuple[], ?> factory = loadCacheFactory( flowProcess ); CascadingCache<Tuple, Tuple[]> cache = factory.create( flowProcess ); cache.setCacheEvictionCallback( new Eviction() ); Integer cacheCapacity = capacity; if( capacity == 0 ) cacheCapacity = getCacheCapacity( flowProcess ); cache.setCapacity( cacheCapacity.intValue() ); cache.initialize(); context.lru = cache; operationCall.setContext( context ); } protected Integer getCacheCapacity( FlowProcess flowProcess ) { return getCacheCapacity( flowProcess, COMPOSITE_FUNCTION_CAPACITY, COMPOSITE_FUNCTION_DEFAULT_CAPACITY ); } protected BaseCacheFactory<Tuple, Tuple[], ?> loadCacheFactory( FlowProcess flowProcess ) { return loadCacheFactory( flowProcess, COMPOSITE_FUNCTION_CACHE_FACTORY, DEFAULT_CACHE_FACTORY_CLASS ); } protected Integer getCacheCapacity( FlowProcess flowProcess, String property, int defaultValue ) { Integer cacheCapacity = flowProcess.getIntegerProperty( property ); if( cacheCapacity == null ) cacheCapacity = defaultValue; return cacheCapacity; } protected BaseCacheFactory<Tuple, Tuple[], ?> loadCacheFactory( FlowProcess flowProcess, String property, Class<? extends BaseCacheFactory> type ) { FactoryLoader loader = FactoryLoader.getInstance(); BaseCacheFactory<Tuple, Tuple[], ?> factory = loader.loadFactoryFrom( flowProcess, property, type ); if( factory == null ) throw new CascadingException( "unable to load cache factory, please check your '" + property + "' setting." ); return factory; } @Override public void operate( FlowProcess flowProcess, FunctionCall<CompositeFunction.Context> functionCall ) { TupleEntry arguments = functionCall.getArguments(); Tuple key = TupleHasher.wrapTuple( this.tupleHasher, arguments.selectTupleCopy( groupingFields ) ); Context context = functionCall.getContext(); Tuple[] functorContext = context.lru.get( key ); if( functorContext == null ) { functorContext = new Tuple[ coFunctions.length ]; context.lru.put( key, functorContext ); incrementNumKeysMissed( flowProcess ); } else { incrementNumKeysHit( flowProcess ); } for( int i = 0; i < coFunctions.length; i++ ) { TupleViews.reset( context.arguments[ i ].getTuple(), arguments.getTuple() ); functorContext[ i ] = coFunctions[ i ].aggregate( flowProcess, context.arguments[ i ], functorContext[ i ] ); } } protected void incrementNumKeysFlushed( FlowProcess flowProcess ) { flowProcess.increment( Cache.Num_Keys_Flushed, 1 ); } protected void incrementNumKeysHit( FlowProcess flowProcess ) { flowProcess.increment( Cache.Num_Keys_Hit, 1 ); } protected void incrementNumKeysMissed( FlowProcess flowProcess ) { flowProcess.increment( Cache.Num_Keys_Missed, 1 ); } @Override public void flush( FlowProcess flowProcess, OperationCall<CompositeFunction.Context> operationCall ) { TupleEntryCollector collector = ( (FunctionCall) operationCall ).getOutputCollector(); Tuple result = operationCall.getContext().result; Map<Tuple, Tuple[]> context = operationCall.getContext().lru; for( Map.Entry<Tuple, Tuple[]> entry : context.entrySet() ) completeFunctors( flowProcess, collector, result, entry ); context.clear(); } @Override public void cleanup( FlowProcess flowProcess, OperationCall<Context> operationCall ) { operationCall.setContext( null ); } private void completeFunctors( FlowProcess flowProcess, TupleEntryCollector outputCollector, Tuple result, Map.Entry<Tuple, Tuple[]> entry ) { Tuple[] results = new Tuple[ coFunctions.length + 1 ]; results[ 0 ] = entry.getKey(); Tuple[] values = entry.getValue(); for( int i = 0; i < coFunctions.length; i++ ) results[ i + 1 ] = coFunctions[ i ].complete( flowProcess, values[ i ] ); TupleViews.reset( result, results ); outputCollector.add( result ); } @Override public boolean equals( Object object ) { if( this == object ) return true; if( !( object instanceof CompositeFunction ) ) return false; if( !super.equals( object ) ) return false; CompositeFunction that = (CompositeFunction) object; if( !Arrays.equals( argumentFields, that.argumentFields ) ) return false; if( !Arrays.equals( functorFields, that.functorFields ) ) return false; if( !Arrays.equals( coFunctions, that.coFunctions ) ) return false; if( groupingFields != null ? !groupingFields.equals( that.groupingFields ) : that.groupingFields != null ) return false; return true; } @Override public int hashCode() { int result = super.hashCode(); result = 31 * result + ( groupingFields != null ? groupingFields.hashCode() : 0 ); result = 31 * result + ( argumentFields != null ? Arrays.hashCode( argumentFields ) : 0 ); result = 31 * result + ( functorFields != null ? Arrays.hashCode( functorFields ) : 0 ); result = 31 * result + ( coFunctions != null ? Arrays.hashCode( coFunctions ) : 0 ); return result; } }