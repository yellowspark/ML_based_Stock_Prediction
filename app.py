# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request, send_from_directory,redirect,flash,session,url_for
from flask_login import logout_user,current_user, login_user, login_required
from models import User, Investment, db
from random import random as rae
from datetime import datetime
import train_models as tm
from flask import Markup
import planner 
import utils
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from config import Config

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

app.jinja_env.add_extension('jinja2.ext.do')
#
# @app.route('/favicon.ico')
# def favicon():
#     return send_from_directory(os.path.join(app.root_path, 'static'),
#                           'favicon.ico',mimetype='image/vnd.microsoft.icon')

app.config.from_object(Config)
db.init_app(app)
migrate = Migrate(app, db)
login = LoginManager(app)
login.login_view = 'login'


def perform_training(stock_name, df, models_list):
    all_colors = {'SVR_linear': '#FF9EDD',
                  'SVR_poly': '#FFFD7F',
                  'SVR_rbf': '#FFA646',
                  'linear_regression': '#CC2A1E',
                  'random_forests': '#8F0099',
                  'KNN': '#CCAB43',
                  'elastic_net': '#CFAC43',
                  'DT': '#85CC43',}

   
    dates, prices, ml_models_outputs, prediction_date, test_price = tm.train_predict_plot(stock_name, df, models_list)
    origdates = dates
    if len(dates) > 20:
        dates = dates[-20:]
        prices = prices[-20:]

    all_data = []
    all_data.append((prices, 'false', 'Data', '#000000'))
    for model_output in ml_models_outputs:
        if len(origdates) > 20:
            all_data.append(
                (((ml_models_outputs[model_output])[0])[-20:], "true", model_output, all_colors[model_output]))
        else:
            all_data.append(
                (((ml_models_outputs[model_output])[0]), "true", model_output, all_colors[model_output]))

    all_prediction_data = []
    all_test_evaluations = []
    all_prediction_data.append(("Original", test_price))
    for model_output in ml_models_outputs:
        all_prediction_data.append((model_output, (ml_models_outputs[model_output])[1]))
        all_test_evaluations.append((model_output, (ml_models_outputs[model_output])[2]))

    return all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations

all_files = utils.read_all_stock_files('individual_stocks_5yr')


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method=='POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            user = User.query.filter_by(username=username).first()
            if user is None or not user.check_password(password):
                flash('Invalid username or password','danger')
                return redirect(url_for('login'))
            login_user(user, remember=True)
            return redirect(url_for('landing_function'))
    return render_template('login.html', title='Sign In')

@app.route('/register',methods=['GET', 'POST'])
def register():
    if request.method=='POST':
        email = request.form.get('email')
        username = request.form.get('username')
        cpassword = request.form.get('cpassword')
        password = request.form.get('password')
        # print(cpassword, password, cpassword==password)
        if username and password and cpassword and email:
            if cpassword != password:
                flash('Password do not match','danger')
                return redirect('/register')
            else:
                if User.query.filter_by(email=email).first() is not None:
                    flash('Please use a different email address','danger')
                    return redirect('/register')
                elif User.query.filter_by(username=username).first() is not None:
                    flash('Please use a different username','danger')
                    return redirect('/register')
                else:
                    user = User(username=username, email=email)
                    user.set_password(password)
                    db.session.add(user)
                    db.session.commit()
                    flash('Congratulations, you are now a registered user!','success')
                    return redirect(url_for('login'))
        else:
            flash('Fill all the fields','danger')
            return redirect('/register')

    return render_template('register.html', title='Sign Up page')

@app.route('/forgot',methods=['GET', 'POST'])
def forgot():
    if request.method=='POST':
        email = request.form.get('email')
        if email:
            pass
    return render_template('forgot.html', title='Password reset page')
    
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('landing_function'))

@login_required
@app.route('/user/<username>')
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    investments = Investment.query.filter_by(user_id=user.id).all()


    return render_template('profile.html', user=user, title=f'{user.username} profile', investments=investments)

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    if request.method=='POST':
        current_user.username = request.form.get('username')
        current_user.about_me = request.form.get('aboutme')
        db.session.commit()
        flash('Your changes have been saved.','success')
        return redirect(url_for('edit_profile'))
    return render_template('edit_profile.html', title='Edit Profile',user=user)


@app.before_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()

@app.route('/', methods=['GET', 'POST'])
@app.route('/prediction')
@login_required
def landing_function():
    # all_files = utils.read_all_stock_files('individual_stocks_5yr')
    # df = all_files['A']
    # # df = pd.read_csv('GOOG_30_days.csv')
    # all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data = perform_training('A', df, ['SVR_linear'])
    stock_files = list(all_files.keys())

    return render_template('index.html',show_results="false", stocklen=len(stock_files), stock_files=stock_files, len2=len([]),
                           all_prediction_data=[],
                           prediction_date="", dates=[], all_data=[], len=len([]))

@app.route('/process')
def process_function():
    return redirect('/prediction')

@app.route('/process', methods=['POST'])
def process():

    stock_file_name = request.form['stockfile']
    ml_algoritms = request.form.getlist('mlalgos')

    # all_files = utils.read_all_stock_files('individual_stocks_5yr')
    df = all_files[str(stock_file_name)]
    # df = pd.read_csv('GOOG_30_days.csv')
    all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations = perform_training(str(stock_file_name), df, ml_algoritms)
    stock_files = list(all_files.keys())
    r = True if rae() > .5 else False   
    return render_template('index.html',all_test_evaluations=all_test_evaluations, show_results="true", stocklen=len(stock_files), stock_files=stock_files,
                           len2=len(all_prediction_data),r=r,
                           all_prediction_data=all_prediction_data,
                           prediction_date=prediction_date, dates=dates, all_data=all_data, len=len(all_data),sname=stock_file_name)

@app.route('/invest', methods=['GET', 'POST'])
def invest():
    c = request.args.get('c')
    s = request.args.get('s')
    df = all_files[str(s)]
    if df is None:
        return redirect('/prediction')
    last_record = df.iloc[-1].to_dict()
    price = last_record['open']
    session['price'] = price
    einvest = Investment.query.filter_by(user_id=current_user.id).filter_by(stock=s).first()
    if request.method=='POST':
        stock = request.form.get('stock')
        p = request.form.get('price')
        p = int(p)
        if p > 0:
            if einvest is None:
                inv=Investment(user_id=current_user.id, stock=stock, price=p)
                db.session.add(inv)
            else:
                einvest.price += p
            db.session.commit()
            flash('Your investment has been added','success')
            return redirect(url_for('user', username=current_user.username))
    return render_template('invest.html', title='Invest',c=c,s=s,last_record=last_record,price=price,einvest=einvest)

@app.route('/sell',methods=['POST'])
def sell():
    s = request.form.get('stock')
    p = request.form.get('price')
    einvest = Investment.query.filter_by(user_id=current_user.id).filter_by(stock=s).first()
    if einvest is not None:
        if einvest.price > int(p):
            einvest.price -= int(p)
            einvest.last_action = 'sell'
            db.session.commit()
            flash('Your investment has been sold','success')
            return redirect(url_for('user', username=current_user.username))
        else:
            flash('You cannot sell more than you have','danger')
            return redirect(url_for('user', username=current_user.username))
    else:
        flash('You do not have this stock','danger')
    return redirect(url_for('user', username=current_user.username))

@app.route('/sip', methods=['GET','POST'])
@login_required
def sip():
    if request.method == "POST":
        amount = request.form['amount']
        tenure = request.form['tenure']
        interest = request.form['interest']
        out = planner.sip(int(amount), int(tenure), float(interest),show_amount_list=True)
        return render_template('sip.html', sip=out,amt= amount,rate=interest,tenure=tenure)
    return render_template('sip.html')

@app.route('/emi', methods=['GET','POST'])
@login_required
def emi():
    if request.method == "POST":
        amount = request.form['amount']
        tenure = request.form['tenure']
        interest = request.form['interest']
        out = planner.emi(int(amount), int(tenure), float(interest))
        print(out)
        return render_template('emi.html', emi=out,amt= amount,rate=interest,tenure=tenure)
    return render_template('emi.html')
# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True)
