public class AggregateByLocally extends SubAssembly { public static final int USE_DEFAULT_THRESHOLD = 0; public enum Cache { Num_Keys_Flushed, Num_Keys_Hit, Num_Keys_Missed } public interface Functor extends cascading.operation.CompositeFunction.CoFunction { } public static class CompositeFunction extends cascading.operation.CompositeFunction { public CompositeFunction( Fields groupingFields, Fields argumentFields, CoFunction coFunction, int capacity ) { super( groupingFields, argumentFields, coFunction, capacity ); } public CompositeFunction( Fields groupingFields, Fields[] argumentFields, CoFunction[] coFunctions, int capacity ) { super( groupingFields, argumentFields, coFunctions, capacity ); } protected void incrementNumKeysFlushed( FlowProcess flowProcess ) { flowProcess.increment( AggregateByLocally.Cache.Num_Keys_Flushed, 1 ); } protected void incrementNumKeysHit( FlowProcess flowProcess ) { flowProcess.increment( AggregateByLocally.Cache.Num_Keys_Hit, 1 ); } protected void incrementNumKeysMissed( FlowProcess flowProcess ) { flowProcess.increment( AggregateByLocally.Cache.Num_Keys_Missed, 1 ); } protected Integer getCacheCapacity( FlowProcess flowProcess ) { return getCacheCapacity( flowProcess, AggregateByLocallyProps.AGGREGATE_LOCALLY_BY_CAPACITY, AggregateByLocallyProps.AGGREGATE_LOCALLY_BY_DEFAULT_CAPACITY ); } protected BaseCacheFactory<Tuple, Tuple[], ?> loadCacheFactory( FlowProcess flowProcess ) { return loadCacheFactory( flowProcess, AggregateByLocallyProps.AGGREGATE_LOCALLY_BY_CACHE_FACTORY, AggregateByLocallyProps.DEFAULT_CACHE_FACTORY_CLASS ); } } private String name; private int capacity; private Fields groupingFields; private Fields[] argumentFields; private Functor[] functors; protected AggregateByLocally( String name, int capacity ) { this.name = name; this.capacity = capacity; } protected AggregateByLocally( Fields argumentFields, Functor functor ) { this.argumentFields = Fields.fields( argumentFields ); this.functors = new Functor[]{functor}; } @ConstructorProperties({"pipe", "groupingFields", "assemblies"}) public AggregateByLocally( Pipe pipe, Fields groupingFields, AggregateByLocally... assemblies ) { this( null, pipe, groupingFields, 0, assemblies ); } @ConstructorProperties({"pipe", "groupingFields", "capacity", "assemblies"}) public AggregateByLocally( Pipe pipe, Fields groupingFields, int capacity, AggregateByLocally... assemblies ) { this( null, pipe, groupingFields, capacity, assemblies ); } @ConstructorProperties({"name", "pipe", "groupingFields", "capacity", "assemblies"}) public AggregateByLocally( String name, Pipe pipe, Fields groupingFields, int capacity, AggregateByLocally... assemblies ) { this( name, capacity ); List<Fields> arguments = new ArrayList<>(); List<Functor> functors = new ArrayList<>(); for( int i = 0; i < assemblies.length; i++ ) { AggregateByLocally assembly = assemblies[ i ]; Collections.addAll( arguments, assembly.getArgumentFields() ); Collections.addAll( functors, assembly.getFunctors() ); } initialize( groupingFields, pipe, arguments.toArray( new Fields[ arguments.size() ] ), functors.toArray( new Functor[ functors.size() ] ) ); } protected AggregateByLocally( String name, Pipe pipe, Fields groupingFields, Fields argumentFields, Functor functor, int capacity ) { this( name, capacity ); initialize( groupingFields, pipe, argumentFields, functor ); } protected void initialize( Fields groupingFields, Pipe pipe, Fields argumentFields, Functor functor ) { initialize( groupingFields, pipe, Fields.fields( argumentFields ), new Functor[]{functor} ); } protected void initialize( Fields groupingFields, Pipe pipe, Fields[] argumentFields, Functor[] functors ) { setPrevious( pipe ); this.groupingFields = groupingFields; this.argumentFields = argumentFields; this.functors = functors; verify(); Fields sortFields = Fields.copyComparators( Fields.merge( this.argumentFields ), this.argumentFields ); Fields argumentSelector = Fields.merge( this.groupingFields, sortFields ); if( argumentSelector.equals( Fields.NONE ) ) argumentSelector = Fields.ALL; CompositeFunction function = new CompositeFunction( this.groupingFields, this.argumentFields, this.functors, capacity ); if( name != null ) pipe = new Pipe( name ); pipe = new Each( pipe, argumentSelector, function, Fields.RESULTS ); setTails( pipe ); } protected void verify() { } public Fields getGroupingFields() { return groupingFields; } public Fields[] getFieldDeclarations() { Fields[] fields = new Fields[ this.functors.length ]; for( int i = 0; i < functors.length; i++ ) fields[ i ] = functors[ i ].getDeclaredFields(); return fields; } protected Fields[] getArgumentFields() { return argumentFields; } protected Functor[] getFunctors() { return functors; } @Property(name = "capacity", visibility = Visibility.PUBLIC) @PropertyDescription("Capacity of the aggregation cache.") @PropertyConfigured(value = AggregateByLocallyProps.AGGREGATE_LOCALLY_BY_CAPACITY, defaultValue = "10000") public int getCapacity() { return capacity; } }