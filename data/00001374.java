public class CountByLocally extends AggregateByLocally { public enum Include { ALL, NO_NULLS, ONLY_NULLS } public static class CountPartials implements Functor { private final Fields declaredFields; private final Include include; private final CoercibleType canonical; public CountPartials( Fields declaredFields ) { this( declaredFields, Include.ALL ); } public CountPartials( Fields declaredFields, Include include ) { if( !declaredFields.isDeclarator() || declaredFields.size() != 1 ) throw new IllegalArgumentException( "declaredFields should declare only one field name" ); if( declaredFields.getType( 0 ) == null ) declaredFields = declaredFields.applyTypes( Long.TYPE ); this.declaredFields = declaredFields; if( include == null ) include = Include.ALL; this.include = include; this.canonical = Coercions.coercibleTypeFor( this.declaredFields.getType( 0 ) ); } @Override public Fields getDeclaredFields() { return declaredFields; } @Override public Tuple aggregate( FlowProcess flowProcess, TupleEntry args, Tuple context ) { if( context == null ) context = new Tuple( 0L ); switch( include ) { case ALL: break; case NO_NULLS: if( Tuples.frequency( args, null ) == args.size() ) return context; break; case ONLY_NULLS: if( Tuples.frequency( args, null ) != args.size() ) return context; break; } context.set( 0, context.getLong( 0 ) + 1L ); return context; } @Override public Tuple complete( FlowProcess flowProcess, Tuple context ) { context.set( 0, canonical.canonical( context.getObject( 0 ) ) ); return context; } } @ConstructorProperties({"countField"}) public CountByLocally( Fields countField ) { super( Fields.ALL, new CountPartials( countField.applyTypes( Long.TYPE ) ) ); } @ConstructorProperties({"countField", "include"}) public CountByLocally( Fields countField, Include include ) { super( Fields.ALL, new CountPartials( countField, include ) ); } @ConstructorProperties({"valueFields", "countField"}) public CountByLocally( Fields valueFields, Fields countField ) { super( valueFields, new CountPartials( countField ) ); } @ConstructorProperties({"valueFields", "countField", "include"}) public CountByLocally( Fields valueFields, Fields countField, Include include ) { super( valueFields, new CountPartials( countField, include ) ); } @ConstructorProperties({"pipe", "groupingFields", "countField"}) public CountByLocally( Pipe pipe, Fields groupingFields, Fields countField ) { this( null, pipe, groupingFields, countField ); } @ConstructorProperties({"pipe", "groupingFields", "countField", "threshold"}) public CountByLocally( Pipe pipe, Fields groupingFields, Fields countField, int threshold ) { this( null, pipe, groupingFields, countField, threshold ); } @ConstructorProperties({"name", "pipe", "groupingFields", "countField"}) public CountByLocally( String name, Pipe pipe, Fields groupingFields, Fields countField ) { this( name, pipe, groupingFields, countField, USE_DEFAULT_THRESHOLD ); } @ConstructorProperties({"name", "pipe", "groupingFields", "countField", "threshold"}) public CountByLocally( String name, Pipe pipe, Fields groupingFields, Fields countField, int threshold ) { super( name, pipe, groupingFields, groupingFields, new CountPartials( countField ), threshold ); } @ConstructorProperties({"pipe", "groupingFields", "countField", "include"}) public CountByLocally( Pipe pipe, Fields groupingFields, Fields countField, Include include ) { this( null, pipe, groupingFields, countField, include ); } @ConstructorProperties({"pipe", "groupingFields", "countField", "include", "threshold"}) public CountByLocally( Pipe pipe, Fields groupingFields, Fields countField, Include include, int threshold ) { this( null, pipe, groupingFields, countField, include, threshold ); } @ConstructorProperties({"name", "pipe", "groupingFields", "countField", "include"}) public CountByLocally( String name, Pipe pipe, Fields groupingFields, Fields countField, Include include ) { this( name, pipe, groupingFields, countField, include, USE_DEFAULT_THRESHOLD ); } @ConstructorProperties({"name", "pipe", "groupingFields", "countField", "include", "threshold"}) public CountByLocally( String name, Pipe pipe, Fields groupingFields, Fields countField, Include include, int threshold ) { super( name, pipe, groupingFields, groupingFields, new CountPartials( countField, include ), threshold ); } @ConstructorProperties({"pipe", "groupingFields", "valueFields", "countField"}) public CountByLocally( Pipe pipe, Fields groupingFields, Fields valueFields, Fields countField ) { this( null, pipe, groupingFields, valueFields, countField, Include.ALL ); } @ConstructorProperties({"pipe", "groupingFields", "valueFields", "countField", "threshold"}) public CountByLocally( Pipe pipe, Fields groupingFields, Fields valueFields, Fields countField, int threshold ) { this( null, pipe, groupingFields, valueFields, countField, threshold ); } @ConstructorProperties({"name", "pipe", "groupingFields", "valueFields", "countField"}) public CountByLocally( String name, Pipe pipe, Fields groupingFields, Fields valueFields, Fields countField ) { this( name, pipe, groupingFields, valueFields, countField, USE_DEFAULT_THRESHOLD ); } @ConstructorProperties({"name", "pipe", "groupingFields", "valueFields", "countField", "threshold"}) public CountByLocally( String name, Pipe pipe, Fields groupingFields, Fields valueFields, Fields countField, int threshold ) { super( name, pipe, groupingFields, valueFields, new CountPartials( countField ), threshold ); } @ConstructorProperties({"pipe", "groupingFields", "valueFields", "countField", "include"}) public CountByLocally( Pipe pipe, Fields groupingFields, Fields valueFields, Fields countField, Include include ) { this( null, pipe, groupingFields, valueFields, countField, include ); } @ConstructorProperties({"pipe", "groupingFields", "valueFields", "countField", "include", "threshold"}) public CountByLocally( Pipe pipe, Fields groupingFields, Fields valueFields, Fields countField, Include include, int threshold ) { this( null, pipe, groupingFields, valueFields, countField, include, threshold ); } @ConstructorProperties({"name", "pipe", "groupingFields", "valueFields", "countField", "include"}) public CountByLocally( String name, Pipe pipe, Fields groupingFields, Fields valueFields, Fields countField, Include include ) { this( name, pipe, groupingFields, valueFields, countField, include, USE_DEFAULT_THRESHOLD ); } @ConstructorProperties({"name", "pipe", "groupingFields", "valueFields", "countField", "include", "threshold"}) public CountByLocally( String name, Pipe pipe, Fields groupingFields, Fields valueFields, Fields countField, Include include, int threshold ) { super( name, pipe, groupingFields, valueFields, new CountPartials( countField, include ), threshold ); } }