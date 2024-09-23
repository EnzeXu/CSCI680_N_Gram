public class AverageBy extends AggregateBy { public enum Include { ALL, NO_NULLS } public static class AveragePartials implements Functor { private final Fields declaredFields; private final Include include; public AveragePartials( Fields declaredFields ) { this.declaredFields = declaredFields; this.include = Include.ALL; } public AveragePartials( Fields declaredFields, Include include ) { this.declaredFields = declaredFields; if( include == null ) include = Include.ALL; this.include = include; } @Override public Fields getDeclaredFields() { Fields sumName = new Fields( AverageBy.class.getPackage().getName() + "." + declaredFields.get( 0 ) + ".sum", Double.class ); Fields countName = new Fields( AverageBy.class.getPackage().getName() + "." + declaredFields.get( 0 ) + ".count", Long.class ); return sumName.append( countName ); } @Override public Tuple aggregate( FlowProcess flowProcess, TupleEntry args, Tuple context ) { if( context == null ) context = Tuple.size( 2 ); if( include == Include.NO_NULLS && args.getObject( 0 ) == null ) return context; context.set( 0, context.getDouble( 0 ) + args.getDouble( 0 ) ); context.set( 1, context.getLong( 1 ) + 1 ); return context; } @Override public Tuple complete( FlowProcess flowProcess, Tuple context ) { return context; } } public static class AverageFinal extends BaseOperation<AverageFinal.Context> implements Aggregator<AverageFinal.Context> { protected static class Context { long nulls = 0L; double sum = 0.0D; long count = 0L; Type type = Double.class; CoercibleType canonical; Tuple tuple = Tuple.size( 1 ); public Context( Fields fieldDeclaration ) { if( fieldDeclaration.hasTypes() ) this.type = fieldDeclaration.getType( 0 ); this.canonical = Coercions.coercibleTypeFor( this.type ); } public Context reset() { nulls = 0L; sum = 0.0D; count = 0L; tuple.set( 0, null ); return this; } public Tuple result() { if( count == 0 && nulls != 0 ) return tuple; tuple.set( 0, canonical.canonical( sum / count ) ); return tuple; } } public AverageFinal( Fields fieldDeclaration ) { super( 2, makeFieldDeclaration( fieldDeclaration ) ); if( !fieldDeclaration.isSubstitution() && fieldDeclaration.size() != 1 ) throw new IllegalArgumentException( "fieldDeclaration may only declare 1 field, got: " + fieldDeclaration.size() ); } private static Fields makeFieldDeclaration( Fields fieldDeclaration ) { if( fieldDeclaration.hasTypes() ) return fieldDeclaration; return fieldDeclaration.applyTypes( Double.class ); } @Override public void prepare( FlowProcess flowProcess, OperationCall<Context> operationCall ) { operationCall.setContext( new Context( getFieldDeclaration() ) ); } @Override public void start( FlowProcess flowProcess, AggregatorCall<Context> aggregatorCall ) { aggregatorCall.getContext().reset(); } @Override public void aggregate( FlowProcess flowProcess, AggregatorCall<Context> aggregatorCall ) { Context context = aggregatorCall.getContext(); TupleEntry arguments = aggregatorCall.getArguments(); if( arguments.getObject( 0 ) == null ) { context.nulls++; return; } context.sum += arguments.getDouble( 0 ); context.count += arguments.getLong( 1 ); } @Override public void complete( FlowProcess flowProcess, AggregatorCall<Context> aggregatorCall ) { aggregatorCall.getOutputCollector().add( aggregatorCall.getContext().result() ); } } @ConstructorProperties({"valueField", "averageField"}) public AverageBy( Fields valueField, Fields averageField ) { super( valueField, new AveragePartials( averageField ), new AverageFinal( averageField ) ); } @ConstructorProperties({"valueField", "averageField", "include"}) public AverageBy( Fields valueField, Fields averageField, Include include ) { super( valueField, new AveragePartials( averageField, include ), new AverageFinal( averageField ) ); } @ConstructorProperties({"pipe", "groupingFields", "valueField", "averageField"}) public AverageBy( Pipe pipe, Fields groupingFields, Fields valueField, Fields averageField ) { this( null, pipe, groupingFields, valueField, averageField, USE_DEFAULT_THRESHOLD ); } @ConstructorProperties({"pipe", "groupingFields", "valueField", "averageField", "threshold"}) public AverageBy( Pipe pipe, Fields groupingFields, Fields valueField, Fields averageField, int threshold ) { this( null, pipe, groupingFields, valueField, averageField, threshold ); } @ConstructorProperties({"name", "pipe", "groupingFields", "valueField", "averageField"}) public AverageBy( String name, Pipe pipe, Fields groupingFields, Fields valueField, Fields averageField ) { this( name, pipe, groupingFields, valueField, averageField, USE_DEFAULT_THRESHOLD ); } @ConstructorProperties({"name", "pipe", "groupingFields", "valueField", "averageField", "threshold"}) public AverageBy( String name, Pipe pipe, Fields groupingFields, Fields valueField, Fields averageField, int threshold ) { this( name, Pipe.pipes( pipe ), groupingFields, valueField, averageField, threshold ); } @ConstructorProperties({"pipes", "groupingFields", "valueField", "averageField"}) public AverageBy( Pipe[] pipes, Fields groupingFields, Fields valueField, Fields averageField ) { this( null, pipes, groupingFields, valueField, averageField, USE_DEFAULT_THRESHOLD ); } @ConstructorProperties({"pipes", "groupingFields", "valueField", "averageField", "threshold"}) public AverageBy( Pipe[] pipes, Fields groupingFields, Fields valueField, Fields averageField, int threshold ) { this( null, pipes, groupingFields, valueField, averageField, threshold ); } @ConstructorProperties({"name", "pipes", "groupingFields", "valueField", "averageField"}) public AverageBy( String name, Pipe[] pipes, Fields groupingFields, Fields valueField, Fields averageField ) { this( name, pipes, groupingFields, valueField, averageField, USE_DEFAULT_THRESHOLD ); } @ConstructorProperties({"name", "pipes", "groupingFields", "valueField", "averageField", "threshold"}) public AverageBy( String name, Pipe[] pipes, Fields groupingFields, Fields valueField, Fields averageField, int threshold ) { super( name, pipes, groupingFields, valueField, new AveragePartials( averageField ), new AverageFinal( averageField ), threshold ); } @ConstructorProperties({"pipe", "groupingFields", "valueField", "averageField", "include"}) public AverageBy( Pipe pipe, Fields groupingFields, Fields valueField, Fields averageField, Include include ) { this( null, pipe, groupingFields, valueField, averageField, include, USE_DEFAULT_THRESHOLD ); } @ConstructorProperties({"pipe", "groupingFields", "valueField", "averageField", "include", "threshold"}) public AverageBy( Pipe pipe, Fields groupingFields, Fields valueField, Fields averageField, Include include, int threshold ) { this( null, pipe, groupingFields, valueField, averageField, include, threshold ); } @ConstructorProperties({"name", "pipe", "groupingFields", "valueField", "averageField", "include"}) public AverageBy( String name, Pipe pipe, Fields groupingFields, Fields valueField, Fields averageField, Include include ) { this( name, pipe, groupingFields, valueField, averageField, include, USE_DEFAULT_THRESHOLD ); } @ConstructorProperties({"name", "pipe", "groupingFields", "valueField", "averageField", "include", "threshold"}) public AverageBy( String name, Pipe pipe, Fields groupingFields, Fields valueField, Fields averageField, Include include, int threshold ) { this( name, Pipe.pipes( pipe ), groupingFields, valueField, averageField, include, threshold ); } @ConstructorProperties({"pipes", "groupingFields", "valueField", "averageField", "include"}) public AverageBy( Pipe[] pipes, Fields groupingFields, Fields valueField, Fields averageField, Include include ) { this( null, pipes, groupingFields, valueField, averageField, include, USE_DEFAULT_THRESHOLD ); } @ConstructorProperties({"pipes", "groupingFields", "valueField", "averageField", "include", "threshold"}) public AverageBy( Pipe[] pipes, Fields groupingFields, Fields valueField, Fields averageField, Include include, int threshold ) { this( null, pipes, groupingFields, valueField, averageField, include, threshold ); } @ConstructorProperties({"name", "pipes", "groupingFields", "valueField", "averageField", "include"}) public AverageBy( String name, Pipe[] pipes, Fields groupingFields, Fields valueField, Fields averageField, Include include ) { this( name, pipes, groupingFields, valueField, averageField, include, USE_DEFAULT_THRESHOLD ); } @ConstructorProperties({"name", "pipes", "groupingFields", "valueField", "averageField", "include", "threshold"}) public AverageBy( String name, Pipe[] pipes, Fields groupingFields, Fields valueField, Fields averageField, Include include, int threshold ) { super( name, pipes, groupingFields, valueField, new AveragePartials( averageField, include ), new AverageFinal( averageField ), threshold ); } }