public class StringCoerce extends Coercions.Coerce<String> { public StringCoerce( Map<Type, Coercions.Coerce> coercions ) { super( coercions ); } @Override public Class<String> getCanonicalType() { return String.class; } @Override public String coerce( Object value ) { if( value == null ) return null; return value.toString(); } }