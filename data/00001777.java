public class ValueTupleDeserializer extends BaseDeserializer < ValueTuple > { public ValueTupleDeserializer ( TupleSerialization . SerializationElementReader elementReader ) { super ( elementReader ) ; setReaders ( elementReader . getTupleSerialization ( ) . getMaskedValueFields ( ) ) ; } @ Override protected ValueTuple createTuple ( ) { return new ValueTuple ( ) ; } }