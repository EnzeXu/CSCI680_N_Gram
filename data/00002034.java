public class CoerceBench { @Param({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"}) int to = 0; Object[] canonicalValues = new Object[]{ JsonNodeFactory.instance.nullNode(), JSONCoercibleType.TYPE.canonical( "{ \"name\":\"John\", \"age\":50, \"car\":null }" ), JSONCoercibleType.TYPE.canonical( "{ \"name\":\"John\", \"age\":50, \"car\":null }" ), JSONCoercibleType.TYPE.canonical( "[ \"Ford\", \"BMW\", \"Fiat\" ]" ), JsonNodeFactory.instance.textNode( "1000" ), JsonNodeFactory.instance.numberNode( (short) 1000 ), JsonNodeFactory.instance.numberNode( (short) 1000 ), JsonNodeFactory.instance.numberNode( 1000 ), JsonNodeFactory.instance.numberNode( 1000 ), JsonNodeFactory.instance.numberNode( 1000L ), JsonNodeFactory.instance.numberNode( 1000L ), JsonNodeFactory.instance.numberNode( 1000.000F ), JsonNodeFactory.instance.numberNode( 1000.000F ), JsonNodeFactory.instance.numberNode( 1000.000D ), JsonNodeFactory.instance.numberNode( 1000.000D ) }; Class[] toTypes = new Class[]{ String.class, String.class, Map.class, List.class, String.class, Short.class, Short.TYPE, Integer.class, Integer.TYPE, Long.class, Long.TYPE, Float.class, Float.TYPE, Double.class, Double.TYPE }; CoercibleType coercibleType = JSONCoercibleType.TYPE; Object canonicalValue; CoercionFrom coercion; Class toType; @Setup public void setup() { canonicalValue = canonicalValues[ to ]; toType = toTypes[ to ]; coercion = coercibleType.to( toType ); } @Benchmark public void baseline( Blackhole bh ) { bh.consume( coercibleType.coerce( canonicalValue, toType ) ); } @Benchmark public void coercionFrom( Blackhole bh ) { bh.consume( coercibleType.to( toType ).coerce( canonicalValue ) ); } @Benchmark public void coercionFromFixed( Blackhole bh ) { bh.consume( coercion.coerce( canonicalValue ) ); } }