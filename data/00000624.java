public class HostListActivityTest { private void mockBindToService(TerminalManager terminalManager) { TerminalManager.TerminalBinder stubBinder = mock(TerminalManager.TerminalBinder.class); when(stubBinder.getService()).thenReturn(terminalManager); shadowOf((Application) ApplicationProvider.getApplicationContext()).setComponentNameAndServiceForBindService(new ComponentName("org.connectbot", TerminalManager.class.getName()), stubBinder); } @Test public void bindsToTerminalManager() { TerminalManager terminalManager = spy(TerminalManager.class); mockBindToService(terminalManager); HostListActivity activity = Robolectric.buildActivity(HostListActivity.class).create().start().get(); Intent serviceIntent = new Intent(activity, TerminalManager.class); Intent actualIntent = shadowOf(activity).getNextStartedService(); assertTrue(actualIntent.filterEquals(serviceIntent)); } }