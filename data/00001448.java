public class FloatCoerce extends NumberCoerce<Float> { protected FloatCoerce( Map<Type, Coercions.Coerce> map ) { super( map ); } @Override public Class<Float> getCanonicalType() { return float.class; } @Override protected Float forNull() { return 0F; } @Override protected Float forBoolean( Boolean f ) { return f ? 1F : 0F; } @Override protected <T> Float parseType( T f ) { return Float.parseFloat( f.toString() ); } @Override protected Float asType( Number f ) { return f.floatValue(); } }