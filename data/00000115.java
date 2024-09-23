public class BinaryMemcachedNodeImpl extends TCPMemcachedNodeImpl { private static final int MAX_GET_OPTIMIZATION_COUNT = 4096; private static final int MAX_SET_OPTIMIZATION_COUNT = 65535; private static final int MAX_SET_OPTIMIZATION_BYTES = 2 * 1024 * 1024; public BinaryMemcachedNodeImpl(SocketAddress sa, SocketChannel c, int bufSize, BlockingQueue<Operation> rq, BlockingQueue<Operation> wq, BlockingQueue<Operation> iq, Long opQueueMaxBlockTimeNs, boolean waitForAuth, long dt, long at, ConnectionFactory fa) { super(sa, c, bufSize, rq, wq, iq, opQueueMaxBlockTimeNs, waitForAuth, dt, at, fa); } @Override protected void optimize() { Operation firstOp = writeQ.peek(); if (firstOp instanceof GetOperation) { optimizeGets(); } else if (firstOp instanceof CASOperation) { optimizeSets(); } } private void optimizeGets() { optimizedOp = writeQ.remove(); if (writeQ.peek() instanceof GetOperation) { OptimizedGetImpl og = new OptimizedGetImpl((GetOperation) optimizedOp); optimizedOp = og; while (writeQ.peek() instanceof GetOperation && og.size() < MAX_GET_OPTIMIZATION_COUNT) { GetOperation o = (GetOperation) writeQ.remove(); if (!o.isCancelled()) { og.addOperation(o); } } optimizedOp.initialize(); assert optimizedOp.getState() == OperationState.WRITE_QUEUED; ProxyCallback pcb = (ProxyCallback) og.getCallback(); getLogger().debug("Set up %s with %s keys and %s callbacks", this, pcb.numKeys(), pcb.numCallbacks()); } } private void optimizeSets() { optimizedOp = writeQ.remove(); if (writeQ.peek() instanceof CASOperation) { OptimizedSetImpl og = new OptimizedSetImpl((CASOperation) optimizedOp); optimizedOp = og; while (writeQ.peek() instanceof StoreOperation && og.size() < MAX_SET_OPTIMIZATION_COUNT && og.bytes() < MAX_SET_OPTIMIZATION_BYTES) { CASOperation o = (CASOperation) writeQ.remove(); if (!o.isCancelled()) { og.addOperation(o); } } optimizedOp.initialize(); assert optimizedOp.getState() == OperationState.WRITE_QUEUED; } } }