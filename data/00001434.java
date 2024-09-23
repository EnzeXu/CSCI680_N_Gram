public class TupleEntryChainIterator extends TupleEntryIterator { Iterator<Tuple>[] iterators; int currentIterator = 0; public TupleEntryChainIterator( Fields fields ) { super( fields ); this.iterators = new Iterator[ 1 ]; } public TupleEntryChainIterator( Fields fields, Iterator<Tuple> iterator ) { this( fields ); this.iterators[ 0 ] = iterator; } public TupleEntryChainIterator( Fields fields, Iterator<Tuple>[] iterators ) { super( fields ); this.iterators = iterators; } public boolean hasNext() { if( iterators.length < currentIterator + 1 ) return false; if( iterators[ currentIterator ].hasNext() ) return true; closeCurrent(); currentIterator++; return iterators.length != currentIterator && hasNext(); } public void reset( Iterator<Tuple> iterator ) { this.currentIterator = 0; this.iterators[ 0 ] = iterator; } public void reset( Iterator<Tuple>[] iterators ) { this.currentIterator = 0; this.iterators = iterators; } public TupleEntry next() { hasNext(); entry.setTuple( iterators[ currentIterator ].next() ); return entry; } public void remove() { iterators[ currentIterator ].remove(); } public void close() { if( iterators.length != currentIterator ) closeCurrent(); } protected void closeCurrent() { close( iterators[ currentIterator ] ); } private void close( Iterator iterator ) { if( iterator instanceof Closeable ) { try { ( (Closeable) iterator ).close(); } catch( IOException exception ) { } } } }