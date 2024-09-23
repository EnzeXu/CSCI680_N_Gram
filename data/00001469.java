public class UnitOfWorkExecutorStrategy implements UnitOfWorkSpawnStrategy { private ExecutorService executor; public List<Future<Throwable>> start( UnitOfWork unitOfWork, int maxConcurrentThreads, Collection<Callable<Throwable>> values ) throws InterruptedException { executor = Executors.newFixedThreadPool( maxConcurrentThreads ); List<Future<Throwable>> futures = new ArrayList<>(); for( Callable<Throwable> value : values ) futures.add( executor.submit( value ) ); executor.shutdown(); return futures; } @Override public boolean isCompleted( UnitOfWork unitOfWork ) { return executor == null || executor.isTerminated(); } @Override public void complete( UnitOfWork unitOfWork, int duration, TimeUnit unit ) throws InterruptedException { if( executor == null ) return; executor.awaitTermination( duration, unit ); } }