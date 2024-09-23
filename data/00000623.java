public class StartupTest { private static final long KEYBOARD_DISMISSAL_DELAY_MILLIS = 1000L; @Rule public final ActivityTestRule<HostListActivity> mActivityRule = new ActivityTestRule<>( HostListActivity.class, false, false); @Before public void makeDatabasePristine() { Context testContext = ApplicationProvider.getApplicationContext(); HostDatabase.resetInMemoryInstance(testContext); mActivityRule.launchActivity(new Intent()); } @Test public void canToggleSoftKeyboardVisibility() { Context testContext = ApplicationProvider.getApplicationContext(); assumeThat(testContext.getResources().getConfiguration().hardKeyboardHidden, equalTo(Configuration.HARDKEYBOARDHIDDEN_YES)); SharedPreferences settings = PreferenceManager.getDefaultSharedPreferences(testContext); SharedPreferences.Editor editor = settings.edit(); boolean wasAlwaysVisible = settings.getBoolean(PreferenceConstants.KEY_ALWAYS_VISIBLE, false); try { editor.putBoolean(PreferenceConstants.KEY_ALWAYS_VISIBLE, true).commit(); startNewLocalConnection(); hideAndShowSoftKeyboard(); } finally { editor.putBoolean(PreferenceConstants.KEY_ALWAYS_VISIBLE, wasAlwaysVisible).commit(); } } @Test public void localConnectionDisconnectFromHostList() { startNewLocalConnection(); onView(withId(R.id.console_flip)).perform(closeSoftKeyboard(), pressBack()); onView(withId(R.id.list)) .check(hasHolderItem(allOf(withHostNickname("Local"), withConnectedHost()))) .perform(actionOnHolderItem( allOf(withHostNickname("Local"), withConnectedHost()), longClick())); onView(withText(R.string.list_host_disconnect)).check(matches(isDisplayed())).perform(click()); onView(withId(R.id.list)).check(hasHolderItem(allOf(withHostNickname("Local"), withDisconnectedHost()))); } @Test public void localConnectionDisconnectConsoleActivity() { startNewLocalConnection(); openActionBarOverflowOrOptionsMenu(ApplicationProvider.getApplicationContext()); onView(withText(R.string.list_host_disconnect)).check(matches(isDisplayed())).perform(click()); onView(withId(R.id.list)).check(hasHolderItem(allOf(withHostNickname("Local"), withDisconnectedHost()))); } @Test public void localConnectionCanDelete() { startNewLocalConnectionAndGoBack("Local"); onView(withId(R.id.list)).perform(actionOnHolderItem(withHostNickname("Local"), longClick())); onView(withText(R.string.list_host_delete)).perform(click()); onView(withText(R.string.delete_pos)).perform(click()); } @Test public void localConnectionCanChangeToRed() { startNewLocalConnectionAndGoBack("RedLocal"); changeColor("RedLocal", R.color.red, R.string.color_red); } @Test public void canScrollTerminal() { startNewLocalConnection(); onView(withId(R.id.terminal_view)) .perform(closeSoftKeyboard(), longClick(), swipeUp(), swipeDown()); } @Test public void addHostThenCancelAndDiscard() { onView(withId(R.id.add_host_button)).perform(click()); onView(withId(R.id.quickconnect_field)).perform(typeText("abandoned"), closeSoftKeyboard(), pressBack()); onView(withText(R.string.discard_host_changes_message)).check(matches(isDisplayed())); onView(withText(R.string.discard_host_button)).perform(click()); onView(withId(R.id.add_host_button)).check(matches(isDisplayed())); } @Test public void addHostThenCancelAndKeepEditing() { onView(withId(R.id.add_host_button)).perform(click()); onView(withId(R.id.quickconnect_field)).perform(typeText("abandoned"), closeSoftKeyboard(), pressBack()); onView(withText(R.string.discard_host_changes_message)).check(matches(isDisplayed())); onView(withText(R.string.discard_host_cancel_button)).perform(click()); onView(withId(R.id.quickconnect_field)).check(matches(isDisplayed())); } private void changeColor(String hostName, @ColorRes int color, @StringRes int stringForColor) { onView(withId(R.id.list)).perform(actionOnHolderItem(withHostNickname(hostName), longClick())); onView(withText(R.string.list_host_edit)).perform(click()); onView(withText(R.string.hostpref_color_title)).perform(click()); onView(withText(stringForColor)).perform(click()); onView(withId(R.id.save)).perform(click()); Resources res = ApplicationProvider.getApplicationContext().getResources(); onView(withId(R.id.list)).check(hasHolderItem(withColoredText(res.getColor(color)))); } private void hideAndShowSoftKeyboard() { onView(withId(R.id.console_flip)).perform(closeSoftKeyboard()); onView(withContentDescription(R.string.image_description_show_keyboard)).perform(click()); onView(withId(R.id.console_flip)).perform(loopMainThreadFor(KEYBOARD_DISMISSAL_DELAY_MILLIS)); onView(withContentDescription(R.string.image_description_hide_keyboard)).perform(click()); onView(withId(R.id.console_flip)).perform(loopMainThreadFor(KEYBOARD_DISMISSAL_DELAY_MILLIS)); onView(withContentDescription(R.string.image_description_show_keyboard)).perform(click()); onView(withId(R.id.console_flip)).perform(pressBack()); } private void startNewLocalConnectionAndGoBack(String name) { startNewLocalConnection(name); onView(withId(R.id.console_flip)).perform(closeSoftKeyboard(), pressBack()); onView(withId(R.id.list)).check(hasHolderItem(withHostNickname(name))); } private void startNewLocalConnection() { startNewLocalConnection("Local"); } private void startNewLocalConnection(String name) { onView(withId(R.id.add_host_button)).perform(click()); onView(withId(R.id.protocol_text)).perform(click()); onView(withText("local")).perform(click()); onView(withId(R.id.quickconnect_field)).perform(typeText(name)); onView(withId(R.id.save)).perform(click()); Intents.init(); try { onView(withId(R.id.list)).perform(actionOnHolderItem( withHostNickname(name), click())); intended(hasComponent(ConsoleActivity.class.getName())); } finally { Intents.release(); } onView(withId(R.id.console_flip)).check(matches( hasDescendant(allOf(isDisplayed(), withId(R.id.terminal_view))))); } public static ViewAction closeSoftKeyboard() { return new ViewAction() { private final ViewAction mCloseSoftKeyboard = new CloseKeyboardAction(); @Override public Matcher<View> getConstraints() { return mCloseSoftKeyboard.getConstraints(); } @Override public String getDescription() { return mCloseSoftKeyboard.getDescription(); } @Override public void perform(final UiController uiController, final View view) { mCloseSoftKeyboard.perform(uiController, view); uiController.loopMainThreadForAtLeast(KEYBOARD_DISMISSAL_DELAY_MILLIS); } }; } public static ViewAction loopMainThreadFor(final long millis) { return new ViewAction() { @Override public Matcher<View> getConstraints() { return isEnabled(); } @Override public String getDescription() { return "Returns an action that loops the main thread for at least " + millis + "ms."; } @Override public void perform(final UiController uiController, final View view) { uiController.loopMainThreadForAtLeast(millis); } }; } }