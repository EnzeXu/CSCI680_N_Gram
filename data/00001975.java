public class TuplePairDeserializer extends BaseDeserializer<TuplePair> { private final TupleInputStream.TupleElementReader[] keyReaders; private final TupleInputStream.TupleElementReader[] sortReaders; public TuplePairDeserializer( TupleSerialization.SerializationElementReader elementReader ) { super( elementReader ); Class[] keyClasses = elementReader.getTupleSerialization().getKeyTypes(); Class[] sortClasses = elementReader.getTupleSerialization().getSortTypes(); if( elementReader.getTupleSerialization().areTypesRequired() ) { if( keyClasses == null ) throw new IllegalStateException( "types are required to perform serialization, grouping declared fields: " + elementReader.getTupleSerialization().getKeyFields() ); if( sortClasses == null ) throw new IllegalStateException( "types are required to perform serialization, sorting declared fields: " + elementReader.getTupleSerialization().getSortFields() ); } keyReaders = HadoopTupleInputStream.getReadersFor( elementReader, keyClasses ); sortReaders = HadoopTupleInputStream.getReadersFor( elementReader, sortClasses ); } public TuplePair deserialize( TuplePair tuple ) throws IOException { if( tuple == null ) tuple = createTuple(); Tuple[] tuples = TuplePair.tuples( tuple ); if( keyReaders == null ) tuples[ 0 ] = inputStream.readUnTyped( tuples[ 0 ] ); else tuples[ 0 ] = inputStream.readWith( keyReaders, tuples[ 0 ] ); if( sortReaders == null ) tuples[ 1 ] = inputStream.readUnTyped( tuples[ 1 ] ); else tuples[ 1 ] = inputStream.readWith( sortReaders, tuples[ 1 ] ); return tuple; } @Override protected TuplePair createTuple() { return new TuplePair(); } }