public class ValueTupleSerializer extends BaseSerializer<ValueTuple> { public ValueTupleSerializer( TupleSerialization.SerializationElementWriter elementWriter ) { super( elementWriter ); setWriters( elementWriter.getTupleSerialization().getMaskedValueFields() ); } }