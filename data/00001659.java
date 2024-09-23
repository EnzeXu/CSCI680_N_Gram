public class MemoryCoGroupClosure extends JoinerClosure { private Collection<Tuple>[] collections; private final int numSelfJoins; private final Tuple emptyTuple; private Tuple joinedTuple = new Tuple(); private Tuple[] joinedTuplesArray; private TupleBuilder joinedBuilder; public MemoryCoGroupClosure( FlowProcess flowProcess, int numSelfJoins, Fields[] groupingFields, Fields[] valueFields ) { super( flowProcess, groupingFields, valueFields ); this.numSelfJoins = numSelfJoins; this.emptyTuple = Tuple.size( groupingFields[ 0 ].size() ); this.joinedTuplesArray = new Tuple[ size() ]; this.joinedBuilder = makeJoinedBuilder( groupingFields ); } @Override public int size() { return Math.max( joinFields.length, numSelfJoins + 1 ); } public void reset( Collection<Tuple>[] collections ) { this.collections = collections; } @Override public Iterator<Tuple> getIterator( int pos ) { if( numSelfJoins != 0 ) return collections[ 0 ].iterator(); else return collections[ pos ].iterator(); } @Override public boolean isEmpty( int pos ) { if( numSelfJoins != 0 ) return collections[ 0 ].isEmpty(); else return collections[ pos ].isEmpty(); } @Override public Tuple getGroupTuple( Tuple keysTuple ) { Tuples.asModifiable( joinedTuple ); for( int i = 0; i < collections.length; i++ ) joinedTuplesArray[ i ] = collections[ i ].isEmpty() ? emptyTuple : keysTuple; joinedTuple = joinedBuilder.makeResult( joinedTuplesArray ); return joinedTuple; } static interface TupleBuilder { Tuple makeResult( Tuple[] tuples ); } private TupleBuilder makeJoinedBuilder( final Fields[] joinFields ) { final Fields[] fields = isSelfJoin() ? new Fields[ size() ] : joinFields; if( isSelfJoin() ) Arrays.fill( fields, 0, fields.length, joinFields[ 0 ] ); return new TupleBuilder() { Tuple result = TupleViews.createComposite( fields ); @Override public Tuple makeResult( Tuple[] tuples ) { return TupleViews.reset( result, tuples ); } }; } }