"""
Viewer classes, part of the Elements.pyGLV
    
Elements.pyGLV (Computer Graphics for Deep Learning and Scientific Visualization)
@Copyright 2021-2022 Dr. George Papagiannakis
    
The classes below are all related to the GUI and Display part of the package

Basic design principles are based on the Decorator Design pattern:
	• https://refactoring.guru/design-patterns/decorator
	• https://github.com/faif/python-patterns/blob/master/patterns/structural/decorator.py
"""

from __future__         import annotations
from abc                import ABC, abstractmethod
from typing             import List, Dict, Any
from collections.abc    import Iterable, Iterator
from sys import platform

import sdl2
import sdl2.ext
from sdl2.keycode import SDLK_ESCAPE,SDLK_w
from sdl2.video import SDL_WINDOWPOS_CENTERED, SDL_WINDOW_ALLOW_HIGHDPI
import OpenGL.GL as gl
from OpenGL.GL import shaders
import imgui
from imgui.integrations.sdl2 import SDL2Renderer

import Elements.pyECSS.System  
import Elements.pyECSS.math_utilities as util
import Elements.pyECSS.Event
from Elements.pyECSS.System import System 
from Elements.pyECSS.Component import BasicTransform
import numpy as np

import Elements.utils.Shortcuts as Shortcuts

class RenderWindow(ABC):
    """
    The Abstract base class of the Viewer GUI/Display sub-system of pyglGA
    based on the Decorator Pattern, this class is "wrapped" by decorators
    in order to provide extra cpapabilities e.g. SDL2 window, context and ImGUI widgets    
    """     
    
    def __init__(self):
        self._eventManager = None
        self._scene = None
    
    #define properties for EventManager, Scene objects
    @property #name
    def eventManager(self):
        """ Get RenderWindow's eventManager """
        return self._eventManager
    @eventManager.setter
    def eventManager(self, value):
        self._eventManager = value
        
    @property #name
    def scene(self):
        """ Get RenderWindow's Scene reference """
        return self._scene
    @scene.setter
    def scene(self, value):
        self._scene = value
    
    @abstractmethod
    def init(self):
        raise NotImplementedError
    
    abstractmethod
    def init_post(self):
        raise NotImplementedError
    
    @abstractmethod
    def display(self):
        raise NotImplementedError
    
    @abstractmethod
    def display_post(self):
        raise NotImplementedError
    
    @abstractmethod
    def shutdown(self):
        raise NotImplementedError
    
    @abstractmethod
    def event_input_process(self, running = True):
        raise NotImplementedError
    
    @abstractmethod
    def accept(self, system: Elements.pyECSS.System, event = None):
        """
        Accepts a class object to operate on the RenderWindow, based on the Visitor pattern.

        :param system: [a System object]
        :type system: [System]
        """
        raise NotImplementedError
    
    @classmethod
    def getClassName(cls):
        return cls.__name__


class SDL2Window(RenderWindow):
    """ The concrete subclass of RenderWindow for the SDL2 GUI API 

    :param RenderWindow: [description]
    :type RenderWindow: [type]
    """
    def __init__(self, windowWidth = None, windowHeight = None, windowTitle = None, scene = None, eventManager = None, openGLversion = 4):
        """Constructor SDL2Window for basic SDL2 parameters

        :param windowWidth: [description], defaults to None
        :type windowWidth: [type], optional
        :param windowHeight: [description], defaults to None
        :type windowHeight: [type], optional
        :param windowTitle: [description], defaults to None
        :type windowTitle: [type], optional
        """
        super().__init__()
        
        self._gWindow = None
        self._gContext = None
        self._gVersionLabel = "None"

        self.openGLversion = openGLversion

        if windowWidth is None:
            self._windowWidth = 1024
        else:
            self._windowWidth = windowHeight

        if windowHeight is None:
            self._windowHeight = 1024
        else:
            self._windowHeight = windowHeight

    
        if windowTitle is None:
            self._windowTitle = "SDL2Window"
        else:
            self._windowTitle = windowTitle
        
        if eventManager is not None and scene is None:
            # in case we are testing without a Scene and just an EventManager
            self.eventManager = eventManager
        
        if scene is not None:
            # set the reference of parent RenderWindow to Scene
            # get the reference to EventManager from Scene.ECSSManager
            self._scene = scene
            self.eventManager = scene.world.eventManager
            
        #OpenGL state variables
        self._wireframeMode = False
       
        self._colorEditor = 0.0, 0.0, 0.0
        self._myCamera = np.identity(4)
             
    @property
    def gWindow(self):
        return self._gWindow
    
    
    @property
    def gContext(self):
        return self._gContext
    
    
    def init(self):
        """
        Initialise an SDL2 RenderWindow, not directly but via the SDL2Decorator
        """
        print(f'{self.getClassName()}: init()')
        
        #SDL_Init for the window initialization
        sdl_not_initialised = sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO | sdl2.SDL_INIT_TIMER)
        if sdl_not_initialised !=0:
            print("SDL2 could not be initialised! SDL Error: ", sdl2.SDL_GetError())
            exit(1)
        
        #setting OpenGL attributes for the GL state
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_FLAGS,
                                 sdl2.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG
                                 )
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_PROFILE_MASK,
                                 sdl2.SDL_GL_CONTEXT_PROFILE_CORE
                                 )
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DOUBLEBUFFER, 1)
        
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_ACCELERATED_VISUAL, 1)
        

        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DEPTH_SIZE, 24)
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_STENCIL_SIZE, 8)   
        sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_MULTISAMPLEBUFFERS, 1)

        
        if self.openGLversion==3:
            print("="*24)
            print("Using OpenGL version 3.2")
            print("="*24)
            sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 3)
            sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 2)    
        else: 
            print("="*24)
            print("Using OpenGL version 4.1")
            print("="*24)
            sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 4)
            sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 1)

                     
            
        # Linux doesn't like SDL_GL_MULTISAMPLESAMPLES
        if platform == "linux" or platform == "linux2":
            pass
        else:
            sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_MULTISAMPLESAMPLES, 16)        
        
        sdl2.SDL_SetHint(sdl2.SDL_HINT_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK, b"1")
        sdl2.SDL_SetHint(sdl2.SDL_HINT_VIDEO_HIGHDPI_DISABLED, b"1")
        
        #creating the SDL2 window
        self._gWindow = sdl2.SDL_CreateWindow(self._windowTitle.encode(), 
                                              sdl2.SDL_WINDOWPOS_CENTERED,
                                              sdl2.SDL_WINDOWPOS_CENTERED,
                                              self._windowWidth,
                                              self._windowHeight,
                                            #   sdl2.SDL_WINDOW_ALLOW_HIGHDPI)
                                              sdl2.SDL_WINDOW_OPENGL,
                                              sdl2.SDL_WINDOW_RESIZABLE)
        if self._gWindow is None:
            print("Window could not be created! SDL Error: ", sdl2.SDL_GetError())
            exit(1)
            
        #create the OpenGL context for rendering into the SDL2Window that was constructed just before
        self._gContext = sdl2.SDL_GL_CreateContext(self._gWindow)
        if self._gContext is None:
            print("OpenGL Context could not be created! SDL Error: ", sdl2.SDL_GetError())
            exit(1)
        sdl2.SDL_GL_MakeCurrent(self._gWindow, self._gContext)
        if sdl2.SDL_GL_SetSwapInterval(1) < 0:
            print("Warning: Unable to set VSync! SDL Error: ", sdl2.SDL_GetError())
            # exit(1)
        #obtain the GL versioning system info
        self._gVersionLabel = f'OpenGL {gl.glGetString(gl.GL_VERSION).decode()} GLSL {gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION).decode()} Renderer {gl.glGetString(gl.GL_RENDERER).decode()}'
        print(self._gVersionLabel)
    
    
    def init_post(self):
        """
        Post init method for SDL2
        this should be ctypiically alled AFTER all other GL contexts have been created
        """
        pass
    
    
    def display(self):
        """
        Main display window method to be called standalone or from within a concrete Decorator
        """
        #GPTODO make background clear color as parameter at class level

            
        gl.glClearColor(*self._colorEditor, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        # gl.glDepthFunc(gl.GL_LEQUAL);
        
        
        
        # gl.glDepthMask(gl.GL_FALSE);  
    
    
        #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        
        #setup some extra GL state flags
        if self._wireframeMode:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            #print(f"SDL2Window:display() set wireframemode: {self._wireframeMode}")
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            #print(f"SDL2Window:display() set wireframemode: {self._wireframeMode}")
            
        #print(f'{self.getClassName()}: display()')
    
    
    def display_post(self):
        """
        To be called at the end of each drawn frame to swap double buffers
        """
        sdl2.SDL_GL_SwapWindow(self._gWindow)
        #print(f'{self.getClassName()}: display_post()')       
    
    
    def shutdown(self):
        """
        Shutdown and cleanup SDL2 operations
        """
        print(f'{self.getClassName()}: shutdown()')
        if (self._gContext and self._gWindow is not None):
            sdl2.SDL_GL_DeleteContext(self._gContext)
            sdl2.SDL_DestroyWindow(self._gWindow)
            sdl2.SDL_Quit()   



    def event_input_process(self):
    #def event_input_process(self, running = True):
        """
        process SDL2 basic events and input
        """
        running = True
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    running = False
            if event.type == sdl2.SDL_QUIT:
                running = False
        return running
    
    def accept(self, system: Elements.pyECSS.System, event = None):
        system.apply2SDLWindow(self, event)


class RenderDecorator(RenderWindow):
    """
    Main Decorator class that wraps a RenderWindow so that all other Decorator classes can dynamically be
    adding layered functionality on top of the wrapee (RenderWindow) e.g. ImGUI widgets etc.

    :param RenderWindow: [description]
    :type RenderWindow: [type]
    """
    def __init__(self, wrapee: RenderWindow):
        super().__init__()
        
        self._wrapeeWindow = wrapee
    
    
    @property
    def wrapeeWindow(self):
        return self._wrapeeWindow
    
    
    def init(self):
        """
        [summary]
        """
        self._wrapeeWindow.init()
        print(f'RenderDecorator: init()')
        
        
    def display(self):
        """
        Main decorator display method
        """
        self._wrapeeWindow.display()
        
        
    def shutdown(self):
        """
        [summary]
        """
        self._wrapeeWindow.shutdown()
        print(f'RenderDecorator: shutdown()')   
        
        
    def event_input_process(self):
        """
        extra decorator method to handle input events
        :param running: [description], defaults to True
        :type running: bool, optional
        """
        return self._wrapeeWindow.event_input_process()
    
    
    def display_post(self):
        """
        Post diplay method after all other display calls have been issued
        """
        self._wrapeeWindow.display_post()
    
    
    def init_post(self):
        """
        Post init method
        this should be ctypiically alled AFTER all other GL contexts have been created, e.g. ImGUI context
        """
        self._wrapeeWindow.init_post()
        
    def accept(self, system: Elements.pyECSS.System, event = None):
        pass
                    
class ImGUIDecorator(RenderDecorator):
    """
    ImGUI decorator

    :param RenderDecorator: [description]
    :type RenderDecorator: [type]
    """
    
    def __init__(self, wrapee: RenderWindow, imguiContext = None):
        super().__init__(wrapee)
        if imguiContext is None:
            self._imguiContext = imgui.create_context()
        else:
            self._imguiContext = imguiContext
        self._imguiRenderer = None
        #setup a simple Event: change to wireframe mode via the GUI
        self._updateWireframe = None
        self._updateCamera = None
        # extra UI elements
        self._wireframeMode = False
        self._changed = False 
        self._checkbox = False 
        self._colorEditor = wrapee._colorEditor
        self._eye = (2.5, 2.5, 2.5)
        self._target = (0.0, 0.0, 0.0) 
        self._up = (0.0, 1.0, 0.0)

        ########## Added bool variable to enable to close the imgui window ###############
        self.showElementsWindow = True
        self.elements_x = 10
        self.elements_y = 30

        # self.collapseShortcutsWindow = True
        # self.show_shortcuts_window = False
        # self.shortcuts_x = 10
        # self.shortcuts_y = 400

        self.graph_x = 560
        self.graph_y = 30

        self.openWindows = [] 

        self.collapseElementsWindow = True
        self.collapseScenegraphVisualizer = True


    def init(self):
        """
        Calls Decoratee init() and also sets up events
        """
        self.wrapeeWindow.init()
        if self._imguiContext is None:
            print("Window could not be created! ImGUI Error: ")
            exit(1)
        else:
            # print("Yay! ImGUI context created successfully")
            pass
        
        # GPTODO here is the issue: SDL2Decorator takes an SDLWindow as wrappee wheras
        # ImGUIDEcorator takes and SDL2Decorator and decorates it!
        if isinstance(self.wrapeeWindow, SDL2Window):   
            self._imguiRenderer = SDL2Renderer(self.wrapeeWindow._gWindow)
        
        #
        # Setting up events that this class is publishing (if the EventManager is present in the decorated wrappee)
        #
        self._updateWireframe = Elements.pyECSS.Event.Event(name="OnUpdateWireframe", id=201, value=None)
        if self._wrapeeWindow.eventManager is not None:
            self._wrapeeWindow.eventManager._events[self._updateWireframe.name] = self._updateWireframe
            self._wrapeeWindow.eventManager._publishers[self._updateWireframe.name] = self
        
        
        self._updateCamera = Elements.pyECSS.Event.Event(name="OnUpdateCamera", id=300, value=None)
        if self._wrapeeWindow.eventManager is not None:
            self._wrapeeWindow.eventManager._events[self._updateCamera.name] = self._updateCamera
            self._wrapeeWindow.eventManager._publishers[self._updateCamera.name] = self
        
        # print(f'{self.getClassName()}: init()') 
        
    def display(self):
        """
        ImGUI decorator display: calls wrapee (RenderWindow::display) as well as extra ImGUI widgets
        """
        self.wrapeeWindow.display()
        gl.glClearColor(*self._colorEditor, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.extra()
        #draw menu bar as a header
        self.menuBar()
        #draw scenegraph tree widget
        self.scenegraphVisualiser()

        #print(f'{self.getClassName()}: display()')
        
    def display_post(self):
        # this is important to draw the ImGUI in full mode and not wireframe!
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        
        # render imgui (after 3D scene and just before the SDL double buffer swap window)
        imgui.render()
        self._imguiRenderer.render(imgui.get_draw_data())


        # call the SDL window window swapping in the end of the scene as final render action
        self.wrapeeWindow.display_post()
        
        
    def extra(self):
        """sample ImGUI widgets to be rendered on a RenderWindow
        """
        imgui.set_next_window_size(300.0, 200.0)

        #start new ImGUI frame context
        imgui.new_frame()
        #demo ImGUI window with all widgets
        # imgui.show_test_window()

        ########## Added bool variable to enable to close the imgui window ###############
        if  self.showElementsWindow:
            #new custom imgui window
            imgui.core.set_next_window_collapsed(not self.collapseElementsWindow)
            self.collapseElementsWindow, self.showElementsWindow = imgui.begin("Elements ImGUI window", True)
            if "ElementsWindow" not in self.openWindows:
                self.openWindows.append("ElementsWindow")

            ###### do this so we can be able to move the window after it was collapsed #########
            #######                         and we re open it                           #########
            if self.collapseElementsWindow:
                imgui.set_window_position(self.elements_x,self.elements_y,imgui.ONCE)
            else:
                imgui.set_window_position(self.elements_x,self.elements_y)

            if "Elements ImGUI window" not in self.openWindows:
                self.openWindows.append("Elements ImGUI window")
            #labels inside the window
            imgui.text("PyImgui + PySDL2 integration successful!")
            imgui.text(self._wrapeeWindow._gVersionLabel)
            
            # populate window with extra UI elements
            imgui.separator()
            imgui.new_line()
            #
            # wireframe Event updates the GL state
            self._changed, self._checkbox = imgui.checkbox("Wireframe", self._wireframeMode)
            if self._changed:
                if self._checkbox is True:
                    self._wireframeMode = True
                    self._updateWireframe.value = self._wireframeMode
                    if self._wrapeeWindow.eventManager is not None:
                        self.wrapeeWindow.eventManager.notify(self, self._updateWireframe) 
                    print(f"wireframe: {self._wireframeMode}")
                if self._checkbox is False:
                    self._wireframeMode = False
                    self._updateWireframe.value = self._wireframeMode
                    if self._wrapeeWindow.eventManager is not None:
                        self.wrapeeWindow.eventManager.notify(self, self._updateWireframe) 
                    print(f"wireframe: {self._wireframeMode}")
            #
            # simple slider for color
            if imgui.tree_node("Background color", imgui.TREE_NODE_OPEN_ON_ARROW):
                self._changed, self._colorEditor = imgui.color_edit3("", *self._colorEditor)
                if self._changed:
                    print(f"_colorEditor: {self._colorEditor}")
                imgui.tree_pop()
            imgui.separator()
            #
            # START
            if imgui.tree_node("TRS", imgui.TREE_NODE_OPEN_ON_ARROW):
                # simple slider for eye - IMPORTANT PART HERE
                self._changed, self._eye = imgui.drag_float3( "Eye", *self._eye, change_speed = 0.01, min_value=-10, max_value=10,format="%.3f")
                if self._changed:
                    self._updateCamera.value = util.lookat(util.vec(self._eye), util.vec(self._target), util.vec(self._up))
                    print ("NEW CAMERA VALUE", self._updateCamera.value)
                    if self._wrapeeWindow.eventManager is not None:
                            self.wrapeeWindow.eventManager.notify(self, self._updateCamera)
                    print(f"_eye: {self._eye}")
                imgui.separator()
                #
                # simple slider for target
                self._changed, self._target = imgui.drag_float3( "Target", *self._target, change_speed = 0.01, min_value=-10, max_value=10,format="%.3f")
                if self._changed:
                    self._updateCamera.value = util.lookat(util.vec(self._eye), util.vec(self._target), util.vec(self._up))
                    print ("NEW CAMERA VALUE", self._updateCamera.value)
                    if self._wrapeeWindow.eventManager is not None:
                        self.wrapeeWindow.eventManager.notify(self, self._updateCamera)
                    print(f"_target: {self._target}")
                imgui.separator()
                # simple slider for up
                self._changed, self._up = imgui.drag_float3( "Up", *self._up, change_speed = 0.01 ,min_value=-5, max_value=5,format="%.3f")
                if self._changed:
                    self._updateCamera.value = util.lookat(util.vec(self._eye), util.vec(self._target), util.vec(self._up))
                    print ("NEW CAMERA VALUE", self._updateCamera.value)
                    if self._wrapeeWindow.eventManager is not None:
                        self.wrapeeWindow.eventManager.notify(self, self._updateCamera)
                    print(f"_up: {self._up}")
                imgui.tree_pop()
            imgui.separator()
            # END
            # simple FPS counter
            if imgui.tree_node("FPS", imgui.TREE_NODE_OPEN_ON_ARROW):
                strFrameRate = str(("Application average: ", imgui.get_io().framerate, " FPS"))
                imgui.text(strFrameRate)
                imgui.tree_pop()
            #end imgui frame context
            imgui.end()
        
        #print(f'{self.getClassName()}: extra()')
    
    def scenegraphVisualiser(self):
        """display the ECSS in an ImGUI tree node structure
        Typically this is a custom widget to be extended in an ImGUIDecorator subclass 
        """
        pass
        
    def accept(self, system: Elements.pyECSS.System, event = None):
        system.apply2ImGUIDecorator(self, event)

    ### After collapsing windows, move all windows to top left corner
    ### Use the top corner to cover all window sizes 
    ### since for now we dont knowt the screen height dynamically
    def align_windows_top_left(self):
        starting_y = 20
        
        self.elements_x = 10
        self.elements_y = starting_y

        starting_y += 20

        if Shortcuts.show_shortcuts_window:
            Shortcuts.shortcuts_x = 10
            Shortcuts.shortcuts_y = starting_y
            starting_y += 20
        if Shortcuts.showGUI_text:
            Shortcuts.GUItext_x = 10
            Shortcuts.GUItext_y = starting_y
            starting_y += 20
        self.graph_x = 10
        self.graph_y = starting_y

        
        
    ######  MENU BAR #########
    def menuBar(self):
        # Create the header bar
        imgui.begin_main_menu_bar()

        # Create the "File" dropdown menu
        if imgui.begin_menu("File", True):
            imgui.menu_item("Save")
            imgui.separator()
            imgui.menu_item("Save as")
            imgui.separator()
            if imgui.menu_item("Minimize")[1]:
                sdl2.SDL_MinimizeWindow(self.wrapeeWindow._gWindow)
            imgui.separator()
            if imgui.menu_item("Exit")[1]:
                exit()
            
            imgui.end_menu()

        # Create the "View" dropdown menu
        if imgui.begin_menu("View"):
            # Add a "Shortcuts" submenu
            if imgui.menu_item("Shortcuts")[1]:
                Shortcuts.show_shortcuts_window = True   
                Shortcuts.displayShortcutsGUI()
            if imgui.menu_item("Elements ImGUI Window")[1]:
                self.showElementsWindow = True           
            if imgui.menu_item("Collapse Windows")[1]:
                self.collapseElementsWindow = False
                Shortcuts.collapseShortcutsWindow = False
                Shortcuts.displayShortcutsGUI()
                Shortcuts.collapseGUI_text = False
                self.collapseScenegraphVisualizer = False
                self.align_windows_top_left()
            imgui.end_menu()

        # Create the "Help" dropdown menu
        if imgui.begin_menu("Help"):
            # Add a "Shortcuts" submenu
            if imgui.menu_item("Example Description")[1]:
                Shortcuts.showGUI_text = True   
            if imgui.menu_item("FAQ")[1]:
                pass
            imgui.end_menu()
        # End the main menu bar
        imgui.end_main_menu_bar()

class ImGUIecssDecorator(ImGUIDecorator):
    
    """custom ImGUI decorator for this example

    :param ImGUIDecorator: [description]
    :type ImGUIDecorator: [type]
    """

    def __init__(self, wrapee: RenderWindow, imguiContext = None):
        super().__init__(wrapee, imguiContext)
        self.selected = None; # Selected should be a component

        self.io = imgui.get_io()
        
        # TRS Variables 
        self.translation = {};
        self.translation["x"] = 0; self.translation["y"] = 0; self.translation["z"] = 0; 

        self.rotation = {};
        self.rotation["x"] = 0; self.rotation["y"] = 0; self.rotation["z"] = 0; 

        self.scale = {};
        self.scale["x"] = 0; self.scale["y"] = 0; self.scale["z"] = 0; 

        self.color = [255, 50, 50];

        self.lctrl = False
        
        self.traverseCamera()
        

    def drawNode(self, node, display_name=None):
        if display_name is None or node.name == display_name:
            # Draw node here
            for child in node.children:
                self.drawNode(child, display_name)
                
    def scenegraphVisualiser(self):
        """display the ECSS in an ImGUI tree node structure
        Typically this is a custom widget to be extended in an ImGUIDecorator subclass
        """
        sceneRoot = self.wrapeeWindow.scene.world.root.name
        if sceneRoot is None:
            sceneRoot = "ECSS Root Entity"

        twoColumn = False

        imgui.core.set_next_window_collapsed(not self.collapseScenegraphVisualizer)

        if twoColumn:
            # 2 Column Version
            self.collapseScenegraphVisualizer, _ = imgui.begin("ECSS graph")
            if "ECSS graph" not in self.openWindows:
                self.openWindows.append("ECSS graph")
            imgui.columns(2, "Properties")
            if imgui.tree_node(sceneRoot, imgui.TREE_NODE_OPEN_ON_ARROW):
                self.drawNode(self.wrapeeWindow.scene.world.root)
                imgui.tree_pop()
            imgui.next_column()
            imgui.text("Properties")
            imgui.separator()
        else:
            self.collapseScenegraphVisualizer, _ = imgui.begin("ECSS graph")
            if "ECSS graph" not in self.openWindows:
                self.openWindows.append("ECSS graph")
            imgui.columns(1, "Properties")
            # below is a recursive call to build-up the whole scenegraph as ImGUI tree
            # if imgui.tree_node(sceneRoot, imgui.TREE_NODE_OPEN_ON_ARROW):
            # self.drawNode(self.wrapeeWindow.scene.world.root)
            # imgui.tree_pop()
            # imgui.next_column()
            imgui.text("Properties")
            imgui.separator()
        
        ###### do this so we can be able to move the window after it was collapsed #########
        #######                         and we re open it                           #########
        if self.collapseScenegraphVisualizer:
            imgui.set_window_position(self.graph_x,self.graph_y,imgui.ONCE)
        else:
            imgui.set_window_position(self.graph_x,self.graph_y)
        # smallerTRSgui = True
        # TRS sample
        # if(isinstance(self.selected, BasicTransform)):

        if imgui.tree_node("Translation", imgui.TREE_NODE_LEAF):
            imgui.same_line() 
            # changed, value = imgui.slider_float("X", self.translation["x"], -3, 3, "%.01f", 1);
            # self.translation["x"] = value;
            # changed, value = imgui.slider_float("Y", self.translation["y"], -3, 3, "%.01f", 1);
            # self.translation["y"] = value;
            # changed, value = imgui.slider_float("Z", self.translation["z"], -3, 3, "%.01f", 1);
            # self.translation["z"] = value;
            changed, value = imgui.drag_float3(
                "X,Y,Z",
                self.translation["x"],
                self.translation["y"],
                self.translation["z"],
                0.01,
                -30,
                30,
                "%.001f",
                1,
            )
            self.translation["x"], self.translation["y"], self.translation["z"] = (
                value[0],
                value[1],
                value[2],
            )
            imgui.tree_pop()
        if imgui.tree_node("Rotation   ", imgui.TREE_NODE_LEAF):
            imgui.same_line() 
            # changed, value = imgui.slider_float("X", self.rotation["x"], -90, 90, "%.1f", 1);
            # self.rotation["x"] = value;
            # changed, value = imgui.slider_float("Y", self.rotation["y"], -90, 90, "%.1f", 1);
            # self.rotation["y"] = value;
            # changed, value = imgui.slider_float("Z", self.rotation["z"], -90, 90, "%.1f", 1);
            # self.rotation["z"] = value;
            changed, value = imgui.drag_float3(
                "X,Y,Z",
                self.rotation["x"],
                self.rotation["y"],
                self.rotation["z"],
                1,
                -180,
                180,
                "%.1f",
                1,
            )
            self.rotation["x"], self.rotation["y"], self.rotation["z"] = (
                value[0],
                value[1],
                value[2],
            )
            imgui.tree_pop()
        if imgui.tree_node("Scale      ", imgui.TREE_NODE_LEAF):
            imgui.same_line() 
            # changed, value = imgui.slider_float("X", self.scale["x"], 0, 3, "%.01f", 1);
            # self.scale["x"] = value;
            # changed, value = imgui.slider_float("Y", self.scale["y"], 0, 3, "%.01f", 1);
            # self.scale["y"] = value;
            # changed, value = imgui.slider_float("Z", self.scale["z"], 0, 3, "%.01f", 1);
            # self.scale["z"] = value;
            changed, value = imgui.drag_float3(
                "X,Y,Z",
                self.scale["x"],
                self.scale["y"],
                self.scale["z"],
                0.01,
                0,
                4,
                "%.01f",
                1,
            )
            self.scale["x"], self.scale["y"], self.scale["z"] = (
                value[0],
                value[1],
                value[2],
            )
            imgui.tree_pop()

        if twoColumn:
            pass
        else:
            imgui.separator()
            if imgui.tree_node(sceneRoot, imgui.TREE_NODE_OPEN_ON_ARROW):
                self.drawNode(self.wrapeeWindow.scene.world.root)
                imgui.tree_pop()

        imgui.end()
    
            
    def drawNode(self, component):
        #create a local iterator of Entity's children
        if component._children is not None:
            debugIterator = iter(component._children)
            #call print() on all children (Concrete Components or Entities) while there are more children to traverse
            done_traversing = False
            while not done_traversing:
                try:
                    comp = next(debugIterator)
                    imgui.indent(10)
                except StopIteration:
                    done_traversing = True
                else:
                    # using ## creates unique labels, without showing anything after ##
                    # see: https://github.com/ocornut/imgui/blob/master/docs/FAQ.md#q-how-can-i-have-multiple-widgets-with-the-same-label
                    if imgui.tree_node(comp.name + "##" + str(comp.id), imgui.TREE_NODE_OPEN_ON_ARROW):
                        imgui.text(comp.name)
                        _, selected = imgui.selectable(comp.__str__(), True)
                        if selected:
                            if comp != self.selected: # First time selecting it. Set trs values to GUI;
                                self.selected = comp;
                                if isinstance(comp, BasicTransform):
                                    [x, y, z] = comp.translation;
                                    self.translation["x"] = x;
                                    self.translation["y"] = y;
                                    self.translation["z"] = z;
                                    [x, y, z] = comp.scale;
                                    self.scale["x"] = x;
                                    self.scale["y"] = y;
                                    self.scale["z"] = z;
                                    [x, y, z] = comp.rotationEulerAngles;
                                    self.rotation["x"] = x;
                                    self.rotation["y"] = y;
                                    self.rotation["z"] = z;
                                # elif isinstance(comp, GameObjectEntity):
                                    # self.color = comp.color.copy();
                            else:                       # Set GUI values to trs;
                                if isinstance(comp, BasicTransform):
                                    transMat = util.translate(self.translation["x"], self.translation["y"], self.translation["z"]);
                                    rotMatX = util.rotate((1, 0, 0), self.rotation["x"])
                                    rotMatY = util.rotate((0, 1, 0), self.rotation["y"])
                                    rotMatZ = util.rotate((0, 0, 1), self.rotation["z"])
                                    scaleMat = util.scale(self.scale["x"], self.scale["y"], self.scale["z"])

                                    comp.trs = util.identity() @ transMat @ rotMatX @ rotMatY @ rotMatZ @ scaleMat;
                                    # comp.trs = scaleMat @ rotMatZ @ rotMatY @ rotMatX @ transMat;
                                elif hasattr(comp, "drawSelfGui"):
                                    comp.drawSelfGui(imgui);

                        imgui.tree_pop()
                    
                    self.drawNode(comp) # recursive call of this method to traverse hierarchy
                    imgui.unindent(10) # Corrent placement of unindent

    
    def traverseCamera(self):
        self.cam = None
        found = False
        rootComp = self.wrapeeWindow.scene.world.root
        if rootComp._children is not None:
            Iterator = iter(rootComp._children)
            done_traversing = False
            while not found and not done_traversing:
                try:
                    comp = next(Iterator)
                except StopIteration:
                    done_traversing = True
                else:
                    if comp.name == "Simple Camera":
                        self.cam = comp
                        found = True
                        
    def updateCamera(self, moveX, moveY, moveZ, rotateX, rotateY):  
        if self.cam != None:
            #for examples 7-11 and pyJANVRED implementations
            cameraspeed = 5
            scaleMat = util.scale(self.scale["x"], self.scale["y"], self.scale["z"])
            combinedMat = scaleMat
            if rotateX or rotateY: 
                rotMatX = util.rotate((1, 0, 0), -self.rotation["y"] * cameraspeed)
                rotMatY = util.rotate((0, 1, 0), self.rotation["x"] * cameraspeed)
                rotMatZ = util.rotate((0, 0, 1), self.rotation["z"] * cameraspeed)
                combinedMat = rotMatX @ rotMatY @ rotMatZ @ combinedMat  
            if moveX or moveY or moveZ:
                transMat = util.translate(self.translation["x"], self.translation["y"], -self.translation["z"])
                combinedMat = transMat @ combinedMat
            self.cam.trans1.trs = self.cam.trans1.trs @ combinedMat
        else:
            #for examples 4-5-6-8-9-10 implementations
            cameraspeed = 0.2
            teye = np.array(self._eye)
            ttarget = np.array(self._target)
            tup = np.array(self._up)

            forwardDir = util.normalise(ttarget - teye)
            rightDir = util.normalise(np.cross(tup, forwardDir))

            eyeUpd = np.array([0.0, 0.0, 0.0])
            targetUpd = np.array([0.0, 0.0, 0.0])   

            if rotateX:
                eyeUpd = rightDir * self.rotation["x"] * cameraspeed
            elif rotateY:
                s,c = util.sincos(1)
                rotDir = util.normalise(util.vec(s, c, 0.0)) * tup
                eyeUpd = rotDir * self.rotation["y"] * cameraspeed
                
            if moveX:
                eyeUpd = -cameraspeed * self.translation["x"] * rightDir
                targetUpd =  eyeUpd
            if moveY:
                eyeUpd = -self.translation["y"] * cameraspeed * tup
                targetUpd = eyeUpd
            if moveZ: 
                eyeUpd =  np.sign(self.translation["z"]) * cameraspeed * forwardDir

            teye += eyeUpd
            ttarget += targetUpd
            if (rotateX or rotateY):
                newForwardDir = util.normalise(ttarget - teye)
                tup = util.normalise(np.cross(newForwardDir, rightDir)) 

            self._eye = tuple(teye)
            self._target = tuple(ttarget)
            self._up = tuple(tup)

            self._updateCamera.value = util.lookat(util.vec(self._eye), util.vec(self._target), util.vec(self._up))
            if self._wrapeeWindow.eventManager is not None:
                self.wrapeeWindow.eventManager.notify(self, self._updateCamera)
        
 
    def on_mouse_motion(self, event, x, y, dx, dy):
        """Called when the mouse is moved.

            event: sdl2.events.SDL_Event, 
            x: horiz coord relative to window, y: vert coord relative to window,
            dx: relative horizontal motion, dy: relative vertical motion
        """
        pass

    def on_mouse_press(self, event, x, y, button, dclick):
        """Called when mouse buttons are pressed.

            event: sdl2.events.SDL_Event, 
            x: horiz coord relative to window, y: vert coord relative to window,
            dx: relative horizontal motion, dy: relative vertical motion
            button: RIGHT - MIDDLE - LEFT
            dclick: True - False if button was double click
        """
        pass

    def resetAll(self):
        self.translation["x"] = 0.0
        self.translation["y"] = 0.0
        self.translation["z"] = 0.0
        self.rotation["x"] = 0.0
        self.rotation["y"] = 0.0
        self.rotation["z"] = 0.0
        self.scale["x"]= 1.0
        self.scale["y"]= 1.0
        self.scale["z"]= 1.0

    def cameraHandling(self, x, y, height, width):
        keystatus = sdl2.SDL_GetKeyboardState(None)
        self.resetAll()

        if keystatus[sdl2.SDL_SCANCODE_LSHIFT]:
            if abs(x) > abs(y):
                self.translation["x"] = x/width*60 #np.sign(event.wheel.x)
                self.updateCamera(True, False, False, False, False)
            else:
                self.translation["y"] =  y/height*60 #np.sign(event.wheel.y)
                self.updateCamera(False, True, False, False, False)
        elif keystatus[sdl2.SDL_SCANCODE_LCTRL] or self.lctrl:
            self.translation["z"] =  y/height*60 #-np.sign(event.wheel.y) 
            self.updateCamera(False, False, True, False, False)
        else:
            if abs(x) > abs(y):
                self.rotation["x"] = np.sign(x) #event.wheel.x/height*180
                self.updateCamera(False, False,False, True, False)
            else:
                self.rotation["y"] = np.sign(y) #event.wheel.y/width*180
                self.updateCamera(False, False,False, False, True)


    ##################  toggle the wireframe #############################
    def toggle_Wireframe(self):
        self._wireframeMode = not self._wireframeMode
        self._updateWireframe.value = self._wireframeMode
        self.wrapeeWindow.eventManager.notify(self, self._updateWireframe)

    def handle_imgui_scroll(self,x,y):
        self._wireframeMode = not self._wireframeMode
        self._updateWireframe.value = self._wireframeMode
        self.wrapeeWindow.eventManager.notify(self, self._updateWireframe)

    def event_input_process(self):
        """
        process SDL2 basic events and input
        """
        #width = self.wrapeeWindow.scene._renderWindow.windowWidth
        #height = self.wrapeeWindow.scene._gContext().windowWidth
        #width = self.wrapeeWindow._gWindow.windowWidth
        running = True
        events = sdl2.ext.get_events()
        width =  1200   #TODO get the window width and height
        height = 800 
        
        ### set up a hot key to easily switch between common keys like shift,ctrl etc
        ### default at left alt
        alt_Key = sdl2.KMOD_ALT
        leftShift_Key = sdl2.KMOD_LSHIFT
        rightShift_Key = sdl2.KMOD_RSHIFT
        ctrl_Key = sdl2.KMOD_CTRL

        shortcut_HotKey = alt_Key

        for event in events:
            
            if event.type == sdl2.SDL_MOUSEWHEEL:
                # # Check if the mouse is hovering over an ImGui window
                # if self.guiHovered:
                #     # Handle scrolling within the ImGui window
                #     self.handle_imgui_scroll(event.wheel.x, event.wheel.y)
                #     continue
                # else:
                x = event.wheel.x
                y = event.wheel.y
                self.cameraHandling(x,y,height,width)
                continue   

            if event.type == sdl2.SDL_MOUSEBUTTONUP:
                pass

            # on_mouse_press
            buttons = event.motion.state
            if buttons & sdl2.SDL_BUTTON_RMASK:
                x = -event.motion.xrel  
                y = event.motion.yrel 
                self.cameraHandling(x, y, height, width)
                
                continue               

            #keyboard events
            if event.type == sdl2.SDL_KEYDOWN:
                
                # if event.key.keysym.sym == sdl2.SDLK_UP or event.key.keysym.sym == sdl2.SDLK_w :
                #     pass
                # if event.key.keysym.sym == sdl2.SDLK_DOWN or event.key.keysym.sym == sdl2.SDLK_s :
                #     pass
                # if event.key.keysym.sym == sdl2.SDLK_LEFT or event.key.keysym.sym == sdl2.SDLK_a :
                #     pass
                # if event.key.keysym.sym == sdl2.SDLK_RIGHT or event.key.keysym.sym == sdl2.SDLK_d :
                #     pass
                ##################  toggle the wireframe using the alt+F buttons  #############################
                if (event.key.keysym.sym == sdl2.SDLK_f and (sdl2.SDL_GetModState() & shortcut_HotKey)):
                    self.toggle_Wireframe()
                
                ########## shortcuts for selected node from the tree ###########
                if self.selected:
                    ################# - translate on x axis when node is selected using W+alt ###########################
                    if(event.key.keysym.sym == sdl2.SDLK_w and (sdl2.SDL_GetModState() & shortcut_HotKey)):
                        self.translation["x"] -= 0.1
                    ################# + translate on x axis when node is selected using W ###########################
                    elif(event.key.keysym.sym == sdl2.SDLK_w):
                        self.translation["x"] += 0.1
                    
                    # ################# - translate on y axis when node is selected using E+alt ###########################
                    if(event.key.keysym.sym == sdl2.SDLK_e and (sdl2.SDL_GetModState() & shortcut_HotKey)):
                        self.translation["y"] -= 0.1
                    ################# + translate on y axis when node is selected using E ###########################
                    elif(event.key.keysym.sym == sdl2.SDLK_e):
                        self.translation["y"] += 0.1 
                    
                    # ################# - translate on z axis when node is selected using R+alt ###########################
                    if(event.key.keysym.sym == sdl2.SDLK_r and (sdl2.SDL_GetModState() & shortcut_HotKey)):
                        self.translation["z"] -= 0.1
                    # ################# + translate on z axis when node is selected using R ###########################
                    elif(event.key.keysym.sym == sdl2.SDLK_r):
                        self.translation["z"] += 0.1
                    

                    # ################# - rotate on x axis when node is selected using T+alt ###########################
                    if(event.key.keysym.sym == sdl2.SDLK_t and (sdl2.SDL_GetModState() & shortcut_HotKey)):
                        self.rotation["x"] -= 0.1
                    # ################# + rotate on x axis when node is selected using T ###########################
                    elif(event.key.keysym.sym == sdl2.SDLK_t):
                        self.rotation["x"] += 0.1
                    
                    # ################# - rotate on y axis when node is selected using Y+alt ###########################
                    if(event.key.keysym.sym == sdl2.SDLK_y and (sdl2.SDL_GetModState() & shortcut_HotKey)):
                        self.rotation["y"] -= 0.1
                    # ################# + rotate on y axis when node is selected using Y ###########################
                    elif(event.key.keysym.sym == sdl2.SDLK_y):
                        self.rotation["y"] += 0.1 
                    
                    # ################# - rotate on z axis when node is selected using U+alt ###########################
                    if(event.key.keysym.sym == sdl2.SDLK_u and (sdl2.SDL_GetModState() & shortcut_HotKey)):
                        self.rotation["z"] -= 0.1
                    # ################# + rotate on z axis when node is selected using U ###########################
                    elif(event.key.keysym.sym == sdl2.SDLK_u):
                        self.rotation["z"] += 0.1
                    
                    ################# scale down on x axis when node is selected using I+alt ###########################
                    if(event.key.keysym.sym == sdl2.SDLK_i  and (sdl2.SDL_GetModState() & shortcut_HotKey)):
                        self.scale["x"] -= 0.1
                    ################# scale up on x axis when node is selected using I ###########################
                    elif(event.key.keysym.sym == sdl2.SDLK_i ):
                        self.scale["x"] += 0.1
                    
                    ################# scale down on y axis when node is selected using O+alt ###########################
                    if(event.key.keysym.sym == sdl2.SDLK_o  and (sdl2.SDL_GetModState() & shortcut_HotKey)):
                        self.scale["y"] -= 0.1
                    ################# scale up on y axis when node is selected using O ###########################
                    elif(event.key.keysym.sym == sdl2.SDLK_o ):
                        self.scale["y"] += 0.1 
                    
                    ################# scale down on z axis when node is selected using P+alt ###########################
                    if(event.key.keysym.sym == sdl2.SDLK_p  and (sdl2.SDL_GetModState() & shortcut_HotKey)):
                        self.scale["z"] -= 0.1
                    ################# scale up on z axis when node is selected using P ###########################
                    elif(event.key.keysym.sym == sdl2.SDLK_p ):
                        self.scale["z"] += 0.1
                    
                if event.key.keysym.sym == sdl2.SDLK_LCTRL:
                    self.lctrl=True
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    running = False

            if event.type == sdl2.SDL_KEYUP and event.key.keysym.sym == sdl2.SDLK_LCTRL:
                self.lctrl = False

            if event.type == sdl2.SDL_QUIT:
                running = False
            #imgui event
            self._imguiRenderer.process_event(event)
            
        #imgui input
        self._imguiRenderer.process_inputs()
        
        return running
        

class RenderGLStateSystem(System):
    """
    System that operates on a RenderDecorator (ImGUIDecorator) and affect GL State

    """
    
    def __init__(self, name=None, type=None, id=None):
        super().__init__(name, type, id)
        
    def update(self):
        """
        method to be subclassed for  behavioral or logic computation 

        """
        pass
    
    
    def apply2ImGUIDecorator(self, imGUIDecorator, event = None):
        """
        method for  behavioral or logic computation 
        when visits Components. 
        
        In this case update GL State from ImGUIDecorator
        
        """
        pass
            
    
    def apply2SDLWindow(self, sdlWindow, event = None):
        """method for  behavioral or logic computation 
        when visits Components. 
        
        In this case update GL State from SDLWindow

        :param sdlWindow: [description]
        :type sdlWindow: [type]
        :param event: [description], defaults to None
        :type event: [type], optional
        """
        if event.name == "OnUpdateWireframe":
            # print(f"RenderGLStateSystem():apply2SDLWindow() actuator system for: {event}")
            sdlWindow._wireframeMode = event.value

        if event.name == "OnUpdateCamera":
            # print(f"OnUpdateCamera: RenderGLStateSystem():apply2SDLWindow() actuator system for: {event}")
            sdlWindow._myCamera = event.value
        
        


if __name__ == "__main__":
    # # The client code.
    gWindow = SDL2Window(openGLversion=3) # uses openGL version 3.2 instead of the default 4.1
    gWindow.init()
    gWindow.init_post()
    running = True
    # MAIN RENDERING LOOP
    while running:
        gWindow.display()
        running = gWindow.event_input_process()
        gWindow.display_post()
    gWindow.shutdown()