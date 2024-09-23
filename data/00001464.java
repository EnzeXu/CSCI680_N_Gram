public class UnitOfWorkDef<T> { private static final Logger LOG = LoggerFactory.getLogger( UnitOfWorkDef.class ); protected String name; protected Set<String> tags = new TreeSet<String>(); public UnitOfWorkDef() { } protected UnitOfWorkDef( UnitOfWorkDef<T> unitOfWorkDef ) { this.name = unitOfWorkDef.name; this.tags.addAll( unitOfWorkDef.tags ); } public String getName() { return name; } public T setName( String name ) { this.name = name; return (T) this; } public String getTags() { return join( tags, "," ); } public T addTag( String tag ) { if( tag == null || tag.isEmpty() ) return (T) this; tag = tag.trim(); if( Util.containsWhitespace( tag ) ) LOG.warn( "tags should not contain whitespace characters: '{}'", tag ); tags.add( tag ); return (T) this; } public T addTags( String... tags ) { for( String tag : tags ) addTag( tag ); return (T) this; } }