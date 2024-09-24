public class TestBuffer extends BaseOperation<TupleEntryCollector> implements Buffer<TupleEntryCollector> { private Tap path; private int expectedSize = -1; private boolean insertHeader; private boolean insertFooter; private Comparable value; private boolean flushCalled = false; public TestBuffer( Tap path, Fields fieldDeclaration, int expectedSize, boolean insertHeader, boolean insertFooter, String value ) { super( fieldDeclaration ); this.path = path; this.expectedSize = expectedSize; this.insertHeader = insertHeader; this.insertFooter = insertFooter; this.value = value; } public TestBuffer( Fields fieldDeclaration, int expectedSize, boolean insertHeader, boolean insertFooter, String value ) { super( fieldDeclaration ); this.expectedSize = expectedSize; this.insertHeader = insertHeader; this.insertFooter = insertFooter; this.value = value; } public TestBuffer( Fields fieldDeclaration, int expectedSize, boolean insertHeader, String value ) { super( fieldDeclaration ); this.expectedSize = expectedSize; this.insertHeader = insertHeader; this.value = value; } public TestBuffer( Fields fieldDeclaration, boolean insertHeader, String value ) { super( fieldDeclaration ); this.insertHeader = insertHeader; this.value = value; } public TestBuffer( Fields fieldDeclaration, Comparable value ) { super( fieldDeclaration ); this.value = value; } public TestBuffer( Fields fieldDeclaration ) { super( fieldDeclaration ); } @Override public void prepare( FlowProcess flowProcess, OperationCall<TupleEntryCollector> operationCall ) { if( path == null ) return; try { operationCall.setContext( flowProcess.openTapForWrite( path ) ); } catch( IOException exception ) { exception.printStackTrace(); } } @Override public void cleanup( FlowProcess flowProcess, OperationCall<TupleEntryCollector> operationCall ) { if( !flushCalled ) throw new RuntimeException( "flush never called" ); if( path == null ) return; operationCall.getContext().close(); } public void operate( FlowProcess flowProcess, BufferCall<TupleEntryCollector> bufferCall ) { if( bufferCall.getJoinerClosure() != null ) throw new IllegalStateException( "joiner closure should be null" ); if( insertHeader ) bufferCall.getOutputCollector().add( new Tuple( value ) ); Iterator<TupleEntry> iterator = bufferCall.getArgumentsIterator(); while( iterator.hasNext() ) { TupleEntry arguments = iterator.next(); if( expectedSize != -1 && arguments.size() != expectedSize ) throw new RuntimeException( "arguments wrong size" ); if( path != null ) bufferCall.getContext().add( arguments ); if( value != null ) bufferCall.getOutputCollector().add( new Tuple( value ) ); else bufferCall.getOutputCollector().add( arguments ); } if( insertFooter ) bufferCall.getOutputCollector().add( new Tuple( value ) ); iterator.hasNext(); } @Override public void flush( FlowProcess flowProcess, OperationCall<TupleEntryCollector> tupleEntryCollectorOperationCall ) { flushCalled = true; } }