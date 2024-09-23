public class TestGate<Incoming, Outgoing> extends Gate<Incoming, Outgoing> implements Window { private final LinkedList<Incoming> list = new LinkedList<Incoming>(); int count = 0; private int size; public TestGate() { } @Override public void bind( StreamGraph streamGraph ) { super.bind( streamGraph ); size = streamGraph.findAllPreviousFor( this ).length; } @Override public void start( Duct previous ) { } @Override public void receive( Duct previous, int ordinal, Incoming incoming ) { list.add( incoming ); } @Override public synchronized void complete( Duct previous ) { count++; if( count < size ) return; try { Grouping grouping = new Grouping(); grouping.joinIterator = list.listIterator(); next.start( this ); next.receive( this, 0, (Outgoing) grouping ); next.complete( this ); } finally { list.clear(); count = 0; } } @Override public void cleanup() { } }